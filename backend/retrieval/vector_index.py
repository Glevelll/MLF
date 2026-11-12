"""Векторный индекс: локальный (numpy, cosine) и Qdrant (prod) с единым интерфейсом."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from backend.retrieval.filters import match


class VectorIndex(ABC):
    @abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None: ...

    @abstractmethod
    def search(self, vector: np.ndarray, k: int = 10,
               filters: dict[str, Any] | None = None) -> list[tuple[str, float, dict]]: ...

    @abstractmethod
    def count(self) -> int: ...


class InMemoryVectorIndex(VectorIndex):
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.ids: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self.matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, ids, vectors, payloads) -> None:
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if vectors.shape[1] != self.dim:
            raise ValueError(f"ожидалась размерность {self.dim}, получено {vectors.shape[1]}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        self.matrix = np.vstack([self.matrix, vectors / norms])
        self.ids += list(ids)
        self.payloads += list(payloads)

    def search(self, vector, k=10, filters=None):
        if not self.ids:
            return []
        v = np.asarray(vector, dtype=np.float32).ravel()
        v = v / (np.linalg.norm(v) + 1e-12)
        sims = self.matrix @ v
        mask = np.array([match(p, filters) for p in self.payloads])
        sims = np.where(mask, sims, -np.inf)
        top = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i]), self.payloads[i]) for i in top if np.isfinite(sims[i])]

    def count(self) -> int:
        return len(self.ids)

    def save(self, path: Path) -> None:
        np.save(path.with_suffix(".npy"), self.matrix)
        path.with_suffix(".json").write_text(json.dumps({"ids": self.ids, "payloads": self.payloads, "dim": self.dim},
                                                        ensure_ascii=False, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> InMemoryVectorIndex:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        idx = cls(meta["dim"])
        idx.matrix = np.load(path.with_suffix(".npy"))
        idx.ids, idx.payloads = meta["ids"], meta["payloads"]
        return idx


class QdrantVectorIndex(VectorIndex):
    """Обёртка над qdrant-client. Фильтры переводятся в qdrant Filter (must / range / match any)."""

    def __init__(self, client: Any, collection: str, dim: int, recreate: bool = False) -> None:
        from qdrant_client.http import models as qm

        self.qm = qm
        self.client = client
        self.collection = collection
        self.dim = dim
        exists = client.collection_exists(collection)
        if recreate and exists:
            client.delete_collection(collection)
            exists = False
        if not exists:
            client.create_collection(collection, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE))

    def _filter(self, filters: dict[str, Any] | None):
        if not filters:
            return None
        qm = self.qm
        must, must_not = [], []
        for key, cond in filters.items():
            if isinstance(cond, dict):
                if "eq" in cond:
                    must.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=cond["eq"])))
                if "in" in cond:
                    must.append(qm.FieldCondition(key=key, match=qm.MatchAny(any=list(cond["in"]))))
                if "ne" in cond:
                    must_not.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=cond["ne"])))
                if "gte" in cond or "lte" in cond:
                    must.append(qm.FieldCondition(key=key, range=qm.Range(gte=cond.get("gte"), lte=cond.get("lte"))))
            else:
                must.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=cond)))
        return qm.Filter(must=must or None, must_not=must_not or None)

    def add(self, ids, vectors, payloads) -> None:
        import uuid

        points = [self.qm.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, i)), vector=list(map(float, v)),
                                      payload={**p, "_doc_id": i})
                  for i, v, p in zip(ids, np.atleast_2d(vectors), payloads, strict=True)]
        for s in range(0, len(points), 256):
            self.client.upsert(self.collection, points=points[s: s + 256])

    def search(self, vector, k=10, filters=None):
        res = self.client.query_points(self.collection, query=list(map(float, np.ravel(vector))), limit=k,
                                       query_filter=self._filter(filters), with_payload=True)
        out = []
        for p in res.points:
            payload = dict(p.payload or {})
            out.append((payload.pop("_doc_id", str(p.id)), float(p.score), payload))
        return out

    def count(self) -> int:
        return int(self.client.count(self.collection).count)
