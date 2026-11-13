"""
RAG sobre resoluciones CREG (adaptado de lecture10/01_rag_nvidia_nim.ipynb).

  Embeddings : NVIDIA NIM (endpoint compatible con OpenAI)
  Índice     : FAISS IndexFlatIP con vectores normalizados (= similitud coseno)
  Chunking   : semántico (párrafos), igual al de clase

El índice se guarda en data/index/ para no recalcular embeddings en cada arranque.
"""
import json
from pathlib import Path

import faiss
import numpy as np
from openai import OpenAI

from config import EMBEDDING_MODEL, get_client
from datos import ROOT, load_documents

INDEX_DIR = ROOT / "data" / "index"


class EmbeddingService:
    """Cliente pequeño para el endpoint de embeddings de NVIDIA NIM."""

    def __init__(self, client: OpenAI, model: str = EMBEDDING_MODEL):
        self.client = client
        self.model = model

    def embed(self, texts: list[str], input_type: str = "passage", batch: int = 48) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), batch):
            response = self.client.embeddings.create(
                model=self.model,
                input=texts[i:i + batch],
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": "END"},
            )
            vectors.extend(item.embedding for item in response.data)
        vectors = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)


def semantic_chunker(text: str, max_chunk_size: int = 900) -> list[str]:
    """Chunking semántico: agrupa párrafos hasta max_chunk_size caracteres.
    Los párrafos más largos que el límite se cortan por oraciones."""
    paragraphs = []
    for p in (p.strip() for p in text.split("\n\n")):
        if not p:
            continue
        if len(p) <= max_chunk_size:
            paragraphs.append(p)
        else:
            paragraphs.extend(s.strip() + "." for s in p.split(". ") if s.strip())

    chunks, current = [], ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chunk_size:
            chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks


class FAISSVectorStore:
    """Vector store minimalista sobre FAISS, con persistencia en disco."""

    def __init__(self, embedder: EmbeddingService):
        self.embedder = embedder
        self.index = None
        self.chunks: list[dict] = []

    # ── Construcción ──────────────────────────────────────────────────────
    def build(self, documents: list[dict], max_chunk_size: int = 900):
        self.chunks = []
        for doc in documents:
            for i, text in enumerate(semantic_chunker(doc["content"], max_chunk_size)):
                if len(text) < 40:          # descartar fragmentos vacíos o títulos sueltos
                    continue
                self.chunks.append({
                    "chunk_id": f"{doc['id']}_chunk_{i}",
                    "source_doc_id": doc["id"],
                    "title": doc["title"],
                    "url": doc["url"],
                    "content": text,
                })
        print(f"  Generando embeddings para {len(self.chunks)} chunks...")
        vectors = self.embedder.embed([c["content"] for c in self.chunks], input_type="passage")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        print(f"  ✅ {self.index.ntotal} chunks indexados")
        return self

    def save(self, path: Path = INDEX_DIR):
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "creg.faiss"))
        (path / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path = INDEX_DIR):
        self.index = faiss.read_index(str(path / "creg.faiss"))
        self.chunks = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        return self

    # ── Búsqueda ──────────────────────────────────────────────────────────
    def search(self, query: str, k: int = 4, threshold: float = 0.25) -> list[dict]:
        q = self.embedder.embed([query], input_type="query")
        scores, idxs = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx >= 0 and score >= threshold:
                results.append({**self.chunks[idx], "score": float(score)})
        return results


def get_vector_store(rebuild: bool = False) -> FAISSVectorStore:
    """Carga el índice desde disco o lo construye si no existe."""
    store = FAISSVectorStore(EmbeddingService(get_client()))
    if not rebuild and (INDEX_DIR / "creg.faiss").exists():
        return store.load()
    store.build(load_documents()).save()
    return store


if __name__ == "__main__":
    get_vector_store(rebuild=True)
