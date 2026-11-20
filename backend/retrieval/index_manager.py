"""Построение/загрузка RAG-индексов из базы знаний (синглтон ретривера)."""
from __future__ import annotations

import json
import threading
from typing import Any

import numpy as np

from backend.core.config import Settings, get_settings
from backend.data.embeddings import DIM, CaseEmbedder, TabularFeatureExtractor, set_embedder
from backend.data.knowledge_base import contextualize, profile_vector
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.vector_index import InMemoryVectorIndex, QdrantVectorIndex, VectorIndex
from backend.utils.logging import get_logger

log = get_logger("rag")
_lock = threading.Lock()
_retriever: HybridRetriever | None = None


def case_payload(c: dict[str, Any]) -> dict[str, Any]:
    feats = c.get("series_features") or {}
    meta = c.get("meta") or {}
    p = {
        "kind": c["kind"], "task_type": c.get("task_type", "tabular"), "problem_type": c.get("problem_type"),
        "best_model": c.get("best_model"), "hyperparams": c.get("hyperparams") or {}, "metrics": c.get("metrics") or {},
        "target_column": c.get("target_column"), "source": meta.get("source"),
    }
    if c["kind"] == "dataset_case":
        p.update(num_rows=int(feats.get("n_rows", 0)), num_features=int(feats.get("n_features", 0)),
                 missing_rate=float(feats.get("missing_rate", 0.0)), has_categorical=feats.get("n_categorical", 0) > 0)
    else:
        p.update(status=meta.get("status"), outcome=meta.get("outcome"),
                 recommendations=meta.get("recommendations", []), confirmed=bool(meta.get("confirmed", False)))
    return p


def numeric_of(c: dict[str, Any]) -> np.ndarray:
    if c["kind"] == "dataset_case":
        return TabularFeatureExtractor().vector(c.get("series_features") or {})
    return profile_vector(c.get("series_features") or {})


def build_retriever(cases: list[dict[str, Any]], glossary: list[dict[str, Any]] | None = None,
                    settings: Settings | None = None, persist: bool = True) -> HybridRetriever:
    settings = settings or get_settings()
    payloads = [case_payload(c) for c in cases]
    texts = [contextualize(c["kind"], c.get("description", ""), p) for c, p in zip(cases, payloads, strict=True)]
    ids = [c["source_id"] for c in cases]
    bm25 = BM25Index()
    bm25.add(ids, texts, payloads)
    by_kind: dict[str, list[int]] = {}
    for i, c in enumerate(cases):
        by_kind.setdefault(c["kind"], []).append(i)
    numeric = {k: np.vstack([numeric_of(cases[i]) for i in idx]) for k, idx in by_kind.items()}
    embedder = CaseEmbedder(settings.text_embedder)
    if cases:
        embedder.fit(texts, numeric)
    vectors = np.zeros((len(cases), DIM))
    for k, idx in by_kind.items():
        vectors[idx] = embedder.embed(k, numeric[k], [texts[i] for i in idx])
    vindex: VectorIndex
    if settings.vector_backend == "qdrant":  # pragma: no cover - требует Qdrant
        from qdrant_client import QdrantClient

        vindex = QdrantVectorIndex(QdrantClient(url=settings.qdrant_url), "kb_cases", DIM, recreate=True)
    else:
        vindex = InMemoryVectorIndex(DIM)
    if cases:
        vindex.add(ids, vectors, payloads)
    gindex = BM25Index()
    if glossary:
        gindex.add([f"g{i}" for i in range(len(glossary))],
                   [f"{g['term']}. {g['definition']}" for g in glossary],
                   [{"term": g["term"], "definition": g["definition"], "category": g.get("category")} for g in glossary])
    retriever = HybridRetriever(bm25, vindex, embedder, gindex)
    if persist:
        set_embedder(embedder)
        d = settings.index_dir
        d.mkdir(parents=True, exist_ok=True)
        embedder.save(d / "embedder.joblib")
        (d / "docs.json").write_text(json.dumps({"ids": ids, "texts": texts, "payloads": payloads,
                                                 "glossary": glossary or []}, ensure_ascii=False, default=str),
                                     encoding="utf-8")
        if isinstance(vindex, InMemoryVectorIndex):
            vindex.save(d / "vectors")
    log.info("RAG index built: %d cases, %d glossary terms", len(cases), len(glossary or []))
    return retriever


def load_retriever(settings: Settings | None = None) -> HybridRetriever:
    settings = settings or get_settings()
    d = settings.index_dir
    docs_path = d / "docs.json"
    if not docs_path.exists():
        return HybridRetriever(BM25Index(), InMemoryVectorIndex(DIM), CaseEmbedder(settings.text_embedder), BM25Index())
    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    bm25 = BM25Index()
    bm25.add(docs["ids"], docs["texts"], docs["payloads"])
    embedder = CaseEmbedder.load(d / "embedder.joblib")
    set_embedder(embedder)
    vindex: VectorIndex
    if settings.vector_backend == "qdrant":  # pragma: no cover
        from qdrant_client import QdrantClient

        vindex = QdrantVectorIndex(QdrantClient(url=settings.qdrant_url), "kb_cases", DIM)
    else:
        vindex = InMemoryVectorIndex.load(d / "vectors")
    gl = docs.get("glossary", [])
    gindex = BM25Index()
    if gl:
        gindex.add([f"g{i}" for i in range(len(gl))], [f"{g['term']}. {g['definition']}" for g in gl],
                   [{"term": g["term"], "definition": g["definition"], "category": g.get("category")} for g in gl])
    return HybridRetriever(bm25, vindex, embedder, gindex)


def get_retriever() -> HybridRetriever:
    global _retriever
    with _lock:
        if _retriever is None:
            _retriever = load_retriever()
        return _retriever


def set_retriever(r: HybridRetriever | None) -> None:
    global _retriever
    with _lock:
        _retriever = r


async def load_kb_from_db() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from sqlalchemy import select

    from backend.db.models import Glossary, KnowledgeBase
    from backend.db.session import session_scope

    async with session_scope() as s:
        rows = (await s.execute(select(KnowledgeBase))).scalars().all()
        gl = (await s.execute(select(Glossary))).scalars().all()
    cases = [{"source_id": r.source_id, "kind": r.kind, "task_type": r.task_type, "problem_type": r.problem_type,
              "target_column": r.target_column, "series_features": r.series_features, "best_model": r.best_model,
              "hyperparams": r.hyperparams, "metrics": r.metrics, "description": r.description, "meta": r.meta}
             for r in rows]
    glossary = [{"term": g.term, "definition": g.definition, "category": g.category} for g in gl]
    return cases, glossary


async def rebuild_from_db() -> dict[str, int]:
    import asyncio

    cases, glossary = await load_kb_from_db()
    retriever = await asyncio.to_thread(build_retriever, cases, glossary)
    set_retriever(retriever)
    return {"cases": len(cases), "glossary": len(glossary)}
