"""Лексический поиск BM25 (Okapi) с фильтрацией по метаданным."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from backend.retrieval.filters import match

_TOKEN = re.compile(r"[a-zа-яё0-9_]+", re.I)
_STOP = {"и", "в", "на", "с", "по", "не", "из", "для", "есть", "the", "a", "of", "and", "to", "in", "is"}


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if t in _STOP or len(t) < 2:
            continue
        # лёгкий стемминг для русского: обрезаем окончания длинных слов
        out.append(t[:-2] if len(t) > 6 and re.match(r"[а-яё]", t) else t)
    return out


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self.texts: list[str] = []
        self.tfs: list[Counter[str]] = []
        self.doc_len: list[int] = []
        self.df: Counter[str] = Counter()

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: list[str], texts: list[str], payloads: list[dict[str, Any]] | None = None) -> None:
        payloads = payloads or [{} for _ in ids]
        for i, t, p in zip(ids, texts, payloads, strict=True):
            toks = tokenize(t)
            tf = Counter(toks)
            self.ids.append(i)
            self.texts.append(t)
            self.payloads.append(p)
            self.tfs.append(tf)
            self.doc_len.append(len(toks))
            self.df.update(tf.keys())

    def _idf(self, term: str) -> float:
        n, df = len(self.ids), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10, filters: dict[str, Any] | None = None) -> list[tuple[str, float, dict]]:
        if not self.ids:
            return []
        q = tokenize(query)
        avg = sum(self.doc_len) / len(self.doc_len) or 1.0
        scores = []
        for idx, tf in enumerate(self.tfs):
            if not match(self.payloads[idx], filters):
                continue
            s = 0.0
            dl = self.doc_len[idx]
            for term in q:
                f = tf.get(term, 0)
                if f:
                    s += self._idf(term) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avg))
            if s > 0:
                scores.append((self.ids[idx], s, self.payloads[idx]))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def text_of(self, doc_id: str) -> str:
        try:
            return self.texts[self.ids.index(doc_id)]
        except ValueError:
            return ""
