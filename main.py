"""
Mini RAG con FastAPI + PostgreSQL (pgvector) + embeddings locales + Groq.

Flujo:
  1) /ingest: divide el texto en fragmentos, genera embeddings y los guarda.
  2) /ask:    convierte la pregunta en embedding, busca los fragmentos más
              cercanos en pgvector y se los pasa al LLM como contexto.
  3) /stats:  devuelve métricas de la base de datos.
  4) GET /    sirve la UI (static/index.html).

Modelos Groq gratuitos recomendados:
  - llama-3.3-70b-versatile   (mejor calidad, default)
  - llama3-8b-8192            (más rápido)
  - mixtral-8x7b-32768        (contexto largo)
  - gemma2-9b-it              (ligero)
"""
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pgvector.psycopg import register_vector
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Carga automáticamente el archivo .env si existe
load_dotenv()

DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5437/rag_db"
)

# Modelo de embeddings local y gratuito (384 dimensiones).
# La primera vez que se ejecuta, descarga el modelo (~90 MB).
embedder = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI(title="Mini RAG")

# ── CORS (útil si abres la UI desde otro puerto durante desarrollo) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Servir archivos estáticos (CSS, JS, assets futuros) ──
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Ruta raíz → UI ──
@app.get("/", include_in_schema=False)
def serve_ui():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI no encontrada. Crea static/index.html")
    return FileResponse(str(index))


def get_conn():
    conn = psycopg.connect(DB_URL, autocommit=True)
    register_vector(conn)  # permite enviar/recibir vectores de numpy
    return conn


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Divide el texto en fragmentos de ~800 caracteres con solape,
    para que una idea cortada entre dos fragmentos no se pierda."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size].strip())
        start += size - overlap
    return [c for c in chunks if c]


def generate_answer(question: str, context: str) -> str:
    """Genera la respuesta con Groq usando solo el contexto recuperado."""
    # Recargar dotenv forzosamente para tomar cambios de .env sin requerir reinicio manual
    load_dotenv(override=True)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "⚠️ Sin GROQ_API_KEY: solo se muestran los fragmentos recuperados. "
            "Configura tu clave en el archivo .env"
        )

    from groq import Groq

    client = Groq(api_key=api_key)
    model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

    try:
        chat = client.chat.completions.create(
            model=model,
            max_tokens=600,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Responde SOLO con la información del contexto proporcionado. "
                        "Si la respuesta no está en el contexto, di que no la encuentras. "
                        "Cita los fragmentos usados como [1], [2], etc. "
                        "Responde en español de forma clara y directa."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Contexto:\n{context}\n\nPregunta: {question}",
                },
            ],
        )
        return chat.choices[0].message.content
    except Exception as e:
        return f"⚠️ Error de Groq ({model}): {str(e)}"


class IngestRequest(BaseModel):
    source: str  # nombre del documento, por ejemplo "manual.txt"
    text: str


class AskRequest(BaseModel):
    question: str
    top_k: int = 4  # cuántos fragmentos recuperar


@app.post("/ingest")
def ingest(req: IngestRequest):
    chunks = chunk_text(req.text)
    if not chunks:
        raise HTTPException(status_code=400, detail="El texto está vacío")

    embeddings = embedder.encode(chunks)
    with get_conn() as conn:
        for chunk, emb in zip(chunks, embeddings):
            conn.execute(
                "INSERT INTO documents (source, content, embedding) VALUES (%s, %s, %s)",
                (req.source, chunk, emb),
            )
    return {"fragmentos_guardados": len(chunks)}


@app.post("/ask")
def ask(req: AskRequest):
    question_embedding = embedder.encode(req.question)

    # <=> es la distancia coseno de pgvector (menor = más parecido).
    # 1 - distancia = similitud.
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT source, content, 1 - (embedding <=> %s) AS similitud
            FROM documents
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (question_embedding, question_embedding, req.top_k),
        ).fetchall()

    if not rows:
        return {"respuesta": "No hay documentos cargados todavía.", "fuentes": []}

    context = "\n\n".join(f"[{i + 1}] ({r[0]}) {r[1]}" for i, r in enumerate(rows))
    return {
        "respuesta": generate_answer(req.question, context),
        "fuentes": [
            {
                "source": r[0],
                "similitud": round(float(r[2]), 3),
                "fragmento": r[1][:200],
            }
            for r in rows
        ],
    }


@app.get("/stats")
def stats():
    """Devuelve métricas básicas de la base de datos para la UI."""
    try:
        with get_conn() as conn:
            total_frags = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            total_docs  = conn.execute("SELECT COUNT(DISTINCT source) FROM documents").fetchone()[0]
        return {
            "fragmentos": total_frags,
            "documentos": total_docs,
            "llm_model":  os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
            "llm_provider": "Groq",
        }
    except Exception as e:
        return {"fragmentos": 0, "documentos": 0, "llm_model": "openai/gpt-oss-120b", "llm_provider": "Groq", "error": str(e)}
