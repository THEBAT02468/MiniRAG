# 🧠 Mini RAG: Retrieval-Augmented Generation con FastAPI, pgvector y LLM

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docker.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Cloud-f55036)](https://groq.com)

Un sistema **RAG (Retrieval-Augmented Generation)** completo, robusto y de alto rendimiento construido desde cero para indexar y consultar información privada usando búsqueda semántica vectorial y modelos de lenguaje de última generación sin alucinaciones.

Incluye una **UI moderna e interactiva inspirada en Claude** con soporte en tiempo real para ingesta de documentos (Drag & Drop), visor de fuentes y fragmentos citados con métricas de similitud de coseno, y panel de administración.

---

## 📌 ¿Qué hace este proyecto? (Problema que resuelve)

Los modelos de lenguaje (LLMs) tienen limitaciones críticas en entornos corporativos:
1. **Desactualización:** No conocen datos privados o recientes de la organización.
2. **Alucinaciones:** Si desconocen un dato, pueden inventar respuestas erróneas.

Este sistema implementa la arquitectura **RAG** para resolver ambos desafíos:
- Extrae y procesa documentos (`.txt`, manuales, políticas, etc.).
- Los fragmenta estratégicamente con solapamiento (*sliding window chunking*) para preservar el contexto de las ideas.
- Genera representaciones matemáticas multidimensionales (**embeddings** de 384 dimensiones) usando el modelo local `all-MiniLM-L6-v2`.
- Almacena e indexa los vectores en **PostgreSQL utilizando la extensión `pgvector`**.
- Cuando el usuario hace una pregunta, realiza una **búsqueda por similitud de coseno** en la base de datos vectorial para recuperar únicamente los fragmentos más relevantes (`Top-K`).
- Envía el contexto recuperado a un LLM (vía **Groq / LLaMA / GPT-OSS**) con un prompt estricto que **garantiza respuestas fidedignas con citas directas [1], [2] y cero alucinaciones**.

---

## 🏗️ Arquitectura del Sistema

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
[ Usuario ] ◄── Respuesta precisa con citas [1] ─────┘
```

---

## 🛠️ Stack Tecnológico

| Capa | Tecnología | Justificación Técnica |
|---|---|---|
| **Lenguaje** | **Python 3.12** | Estándar de la industria en IA y Ciencia de Datos. |
| **API Backend** | **FastAPI + Uvicorn** | Alta velocidad, asíncrono, tipado estático con Pydantic y documentación OpenAPI automática. |
| **Vector DB** | **PostgreSQL + pgvector** | Base de datos relacional robusta con soporte nativo para indexación vectorial y búsqueda por distancia coseno (`<=>`). |
| **Embeddings** | **Sentence-Transformers (`all-MiniLM-L6-v2`)** | Modelo local y eficiente (384 dimensiones) que evita costos por token de embedding y funciona sin latencia externa. |
| **LLM Inference** | **Groq API (`openai/gpt-oss-120b`)** | Inferencia LPUs ultra-rápida (cero latencia de espera) y compatible con modelos abiertos. |
| **Contenedores** | **Docker & Docker Compose** | Despliegue reproducible e independiente del motor PostgreSQL/pgvector. |
| **Frontend** | **HTML5, CSS moderno & Vanilla JS** | UI personalizada tipo Claude, sin dependencias pesadas, ligera, responsive y con dark/light mode. |

---

## ✨ Características Principales

- ⚡ **Búsqueda Semántica Vectorial:** Recupera información por significado conceptual y no por coincidencia de palabras clave exactas.
- 🎯 **Respuestas Auditables con Citas:** Cada afirmación generada por el LLM incluye referencias a los fragmentos indexados (`[1]`, `[2]`).
- 🛡️ **Control de Alucinaciones:** Prompt de sistema optimizado que responde exclusivamente en base a los datos recuperados; si la respuesta no está en los documentos, el modelo lo indica de forma honesta.
- 📂 **Panel de Ingesta Drag & Drop:** Carga interactiva de documentos desde la interfaz web o mediante endpoints REST.
- 📊 **Inspección de Similitud en Tiempo Real:** Visor colapsable que detalla el fragmento exacto y el porcentaje de coincidencia con la pregunta formulada.
- ⚙️ **Configuración Paramétrica:** Selector de `Top-K` dinámico en la UI para ajustar la profundidad de la recuperación (2, 4, 6 u 8 fragmentos).

---

## 🚀 Instalación y Puesta en Marcha

### Prerrequisitos
- [Python 3.11+](https://www.python.org/)
- [Docker Desktop](https://www.docker.com/)
- Una cuenta gratuita en [Groq Console](https://console.groq.com) para obtener una API key.

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/mini-rag-fastapi.git
cd mini-rag-fastapi
```

### 2. Configurar variables de entorno
Copia la plantilla y configura tu clave de API:
```bash
cp .env.example .env
```
Edita `.env` con tu clave:
```env
GROQ_API_KEY=gsk_tu_clave_de_groq_aqui
LLM_MODEL=openai/gpt-oss-120b
DATABASE_URL=postgresql://postgres:postgres@localhost:5437/rag_db
```

### 3. Levantar la base de datos vectorial (Docker)
```bash
docker compose up -d
```
> Esto iniciará un contenedor PostgreSQL 16 con la extensión `pgvector` preconfigurada y la tabla `documents` creada automáticamente.

### 4. Crear el entorno virtual e instalar dependencias
```bash
python -m venv venv

# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 5. Iniciar la aplicación
```bash
uvicorn main:app --reload --port 8000
```

Accede a:
- 🌐 **Interfaz Web:** [http://localhost:8000](http://localhost:8000)
- 📖 **Documentación Swagger / OpenAPI:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 Endpoints de la API

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/` | Servidor de la interfaz gráfica interactiva. |
| `POST` | `/ingest` | Procesa, genera embeddings y almacena un documento en la base vectorial. |
| `POST` | `/ask` | Ejecuta la búsqueda semántica y genera la respuesta mediante el LLM. |
| `GET` | `/stats` | Métricas del sistema (número de documentos, fragmentos indexados y modelo activo). |
| `GET` | `/docs` | Documentación interactiva Swagger. |

#### Ejemplo de Ingesta (`POST /ingest`):
```json
{
  "source": "politica_empresa.txt",
  "text": "Los empleados tienen derecho a 15 días hábiles de vacaciones remuneradas..."
}
```

#### Ejemplo de Consulta (`POST /ask`):
```json
{
  "question": "¿Cuántos días de vacaciones tienen los empleados?",
  "top_k": 4
}
```

---

## 👨‍💻 Autor
Desarrollado con enfoque en ingeniería de software para IA, arquitecturas de recuperación de información (RAG) y despliegue de modelos generativos en producción.
