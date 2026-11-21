-- Habilita pgvector (la extensión que permite guardar y comparar vectores)
CREATE EXTENSION IF NOT EXISTS vector;

-- Cada fila es un fragmento (chunk) de un documento.
-- La dimensión 384 corresponde al modelo de embeddings all-MiniLM-L6-v2.
-- Si cambias de modelo, cambia también este número.
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384)
);

-- Índice HNSW para búsqueda rápida por distancia coseno
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING hnsw (embedding vector_cosine_ops);
