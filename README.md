# 🧠 Mini RAG: Retrieval-Augmented Generation con FastAPI, pgvector y LLM

![Python](https://img.shields.io/badge/Python-3.12-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-API-009688) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791) ![Docker](https://img.shields.io/badge/Docker-Compose-2496ED) ![Groq](https://img.shields.io/badge/LLM-Groq-orange)

Sistema RAG (Retrieval-Augmented Generation) construido desde cero para indexar y consultar documentos propios mediante búsqueda semántica vectorial y un modelo de lenguaje.

Incluye una interfaz web inspirada en Claude, con carga de documentos (drag & drop), visor de fuentes y fragmentos citados con su similitud de coseno, y un panel con estadísticas de la base vectorial.

> Es un proyecto de portafolio para practicar y demostrar la arquitectura RAG. Las limitaciones conocidas están al final de este documento.

![Interfaz del Mini RAG](docs/screenshot.png)

## 📌 ¿Qué problema resuelve?

Los modelos de lenguaje (LLMs) tienen limitaciones al trabajar con información de una organización:

- **Desactualización:** no conocen datos privados o recientes.
- **Alucinaciones:** si desconocen un dato, pueden inventar una respuesta.

La arquitectura RAG ayuda a mitigar ambos problemas dándole al modelo, en cada pregunta, los fragmentos relevantes de tus documentos:

1. Procesa documentos de texto (`.txt`, `.md`).
2. Los divide en fragmentos con solapamiento (sliding window chunking) para no perder ideas en los cortes.
3. Genera embeddings de 384 dimensiones con el modelo local `all-MiniLM-L6-v2`.
4. Almacena los vectores en PostgreSQL con la extensión `pgvector`.
5. Ante una pregunta, busca por similitud de coseno los fragmentos más relevantes (Top-K).
6. Envía ese contexto a un LLM (vía Groq) con un prompt que lo instruye a responder solo con esa información y a citar los fragmentos como [1], [2]. Si la respuesta no está en los documentos, debe indicarlo.

## 🏗️ Arquitectura del sistema

```
                  ┌──────────────────────────────────────────────┐
                  │                 DOCUMENTOS                   │
                  │   (Manuales, guías, archivos de texto)       │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                            [ Ingesta & Chunking ]
                       (Chunks ~800 chars, Overlap 150)
                                         │
                                         ▼
                             [ SentenceTransformers ]
                              (all-MiniLM-L6-v2)
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │   PostgreSQL + pgvector   │
                           │   (Vector DB en Docker)   │
                           └─────────────┬─────────────┘
                                         │
                                         │  Búsqueda semántica
                                         │  (Similitud de Coseno <=>)
                                         ▼
[ Usuario ] ── Pregunta ──► [ Embeddings ] ──► [ Top-K Chunks ]
                                                     │
                                                     ▼
                                            [ Prompt con Contexto ]
                                                     │
                                                     ▼
                                               [ LLM (Groq) ]
                                                     │
                                                     ▼
[ Usuario ] ◄── Respuesta con citas [1] ─────────────┘
```

## 🛠️ Stack tecnológico

| Capa | Tecnología | Justificación |
|---|---|---|
| Lenguaje | Python 3.12 | Estándar en IA y ciencia de datos. |
| API Backend | FastAPI + Uvicorn | Rápido, con tipado vía Pydantic y documentación OpenAPI automática. |
| Vector DB | PostgreSQL + pgvector | Base relacional conocida, con búsqueda por distancia coseno (`<=>`) e índice HNSW. |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) | Modelo local y liviano (384 dimensiones): sin costo por embedding ni dependencia externa. |
| LLM | Groq API (`openai/gpt-oss-120b`) | Inferencia rápida con modelos abiertos y capa gratuita. |
| Contenedores | Docker & Docker Compose | Base de datos reproducible con un solo comando. |
| Frontend | HTML, CSS y JavaScript vanilla | Interfaz ligera, responsive y con modo oscuro/claro. |

## ✨ Características

- ⚡ **Búsqueda semántica:** recupera por significado y no solo por palabras exactas.
- 🎯 **Respuestas con citas:** cada respuesta referencia los fragmentos usados ([1], [2]) y puedes ver el texto y la similitud de cada uno.
- 🛡️ **Respuestas ancladas al contexto:** el prompt de sistema limita al modelo a la información recuperada. Esto reduce las alucinaciones, pero no las elimina por completo.
- 📂 **Ingesta drag & drop:** carga de documentos desde la interfaz o por la API REST.
- ⚙️ **Top-K configurable:** selector de 2, 4, 6 u 8 fragmentos a recuperar.
- 📊 **Estadísticas:** total de documentos, fragmentos y modelo activo.

## 🚀 Instalación y puesta en marcha

### Prerrequisitos

- Python 3.11+
- Docker Desktop
- Una cuenta gratuita en [Groq Console](https://console.groq.com) para obtener una API key.

### 1. Clonar el repositorio

```bash
git clone https://github.com/THEBAT02468/MiniRAG.git
cd MiniRAG
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
```

Edita `.env` con tu clave:

```env
GROQ_API_KEY=gsk_tu_clave_de_groq_aqui
LLM_MODEL=openai/gpt-oss-120b
DATABASE_URL=postgresql://postgres:postgres@localhost:5437/rag_db
```

### 3. Levantar la base de datos vectorial

```bash
docker compose up -d
```

Inicia un contenedor PostgreSQL 16 con `pgvector` y crea la tabla `documents` automáticamente.

### 4. Crear el entorno virtual e instalar dependencias

```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 5. Iniciar la aplicación

```bash
uvicorn main:app --reload --port 8000
```

- 🌐 Interfaz web: http://localhost:8000
- 📖 Swagger / OpenAPI: http://localhost:8000/docs

## 📡 Endpoints de la API

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/` | Sirve la interfaz web. |
| POST | `/ingest` | Fragmenta un documento, genera embeddings y los guarda. |
| POST | `/ask` | Busca por similitud y genera la respuesta con el LLM. |
| GET | `/stats` | Número de documentos, fragmentos y modelo activo. |
| GET | `/docs` | Documentación Swagger. |

**Ejemplo de ingesta (`POST /ingest`):**

```json
{
  "source": "politica_empresa.txt",
  "text": "Los empleados tienen derecho a 15 días hábiles de vacaciones remuneradas..."
}
```

**Ejemplo de consulta (`POST /ask`):**

```json
{
  "question": "¿Cuántos días de vacaciones tienen los empleados?",
  "top_k": 4
}
```

## ⚠️ Limitaciones conocidas

- Sin autenticación: cualquiera con acceso a la API puede cargar y consultar documentos.
- CORS abierto, pensado solo para desarrollo local.
- El chunking es por número de caracteres, no por estructura semántica del texto.
- Solo admite `.txt` y `.md` (sin PDF ni otros formatos).
- Sin reranking ni búsqueda híbrida (texto completo + vectores).
- Sin pruebas automatizadas ni un conjunto de evaluación para medir la calidad de las respuestas.

## 🧭 Próximos pasos

- Armar un conjunto de evaluación (preguntas con la fuente esperada) para medir la recuperación.
- Añadir reranking y búsqueda híbrida.
- Soportar PDF y chunking por secciones.
- Agregar autenticación y restringir CORS.
- Escribir pruebas con `pytest` y un flujo de CI con GitHub Actions.

## 👨‍💻 Autor

Proyecto de portafolio de **Santiago López (THEBAT02468)** para practicar arquitecturas de recuperación de información (RAG) y la integración de LLMs en aplicaciones. Desarrollado con apoyo de herramientas de IA (Claude Code), que usé para acelerar la implementación mientras definía y revisaba la arquitectura.