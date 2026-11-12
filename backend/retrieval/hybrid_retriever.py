"""Гибридный поиск: BM25 + векторный поиск, объединение Reciprocal Rank Fusion."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.vector_index import VectorIndex


@dataclass
class RetrievedDoc:
    id: str
    score: float
    payload: dict[str, Any]
    ranks: dict[str, int] = field(default_factory=dict)

    def brief(self) -> dict[str, Any]:
        return {"id": self.id, "score": round(self.score, 5), "ranks": self.ranks,
                **{k: v for k, v in self.payload.items() if k not in ("text",)}}


def rrf(rankings: dict[str, list[tuple[str, float, dict]]], k: int = 60,
        weights: dict[str, float] | None = None) -> list[RetrievedDoc]:
    """RRF: score(d) = Σ w_r / (k + rank_r(d))."""
    agg: dict[str, RetrievedDoc] = {}
    for name, results in rankings.items():
        w = (weights or {}).get(name, 1.0)
        for rank, (doc_id, _, payload) in enumerate(results, start=1):
            doc = agg.setdefault(doc_id, RetrievedDoc(doc_id, 0.0, payload))
            doc.score += w / (k + rank)
            doc.ranks[name] = rank
    return sorted(agg.values(), key=lambda d: -d.score)


class HybridRetriever:
    def __init__(self, bm25: BM25Index, vector: VectorIndex | None, embedder: Any | None = None,
                 glossary: BM25Index | None = None, rrf_k: int = 60) -> None:
        self.bm25 = bm25
        self.vector = vector
        self.embedder = embedder
        self.glossary = glossary
        self.rrf_k = rrf_k

    @property
    def size(self) -> int:
        return len(self.bm25)

    def search(self, query_text: str, *, kind: str, numeric: np.ndarray | None = None, k: int = 5,
               filters: dict[str, Any] | None = None, candidates: int = 50) -> list[RetrievedDoc]:
        filters = {**(filters or {}), "kind": kind}
        rankings: dict[str, list] = {"bm25": self.bm25.search(query_text, candidates, filters)}
        if self.vector is not None and self.embedder is not None and self.vector.count():
            qv = self.embedder.embed(kind, numeric, [query_text])[0]
            rankings["vector"] = self.vector.search(qv, candidates, filters)
        fused = rrf(rankings, self.rrf_k)
        for d in fused:
            d.payload = {**d.payload, "text": self.bm25.text_of(d.id)}
        return fused[:k]

    def glossary_search(self, query: str, k: int = 3) -> list[dict[str, Any]]:
        if self.glossary is None:
            return []
        return [{"term": p.get("term"), "definition": p.get("definition"), "score": round(s, 3)}
                for _, s, p in self.glossary.search(query, k)]
