"""Импорт базы знаний и справочника в БД (защита от дубликатов) + построение RAG-индексов.

python scripts/import_data.py [--reset]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import delete, select  # noqa: E402

from backend.core.config import get_settings  # noqa: E402
from backend.db.models import Glossary, KnowledgeBase  # noqa: E402
from backend.db.session import dispose_engine, init_db, session_scope  # noqa: E402
from backend.retrieval.index_manager import load_kb_from_db, numeric_of  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def import_all(reset: bool = False) -> dict[str, int]:
    kb_dir = get_settings().kb_dir
    await init_db()
    cases = read_jsonl(kb_dir / "student_cases.jsonl") + read_jsonl(kb_dir / "dataset_cases.jsonl")
    glossary = read_jsonl(kb_dir / "glossary.jsonl")
    added_cases = added_terms = 0
    async with session_scope() as s:
        if reset:
            await s.execute(delete(KnowledgeBase).where(KnowledgeBase.meta["source"].as_string() != "hitl"))
            await s.execute(delete(Glossary))
        existing = set((await s.execute(select(KnowledgeBase.source_id))).scalars().all())
        for c in cases:
            if c["source_id"] in existing:
                continue
            s.add(KnowledgeBase(**{k: c.get(k) for k in ("source_id", "kind", "task_type", "problem_type",
                                                         "target_column", "series_features", "best_model",
                                                         "hyperparams", "metrics", "description", "meta")}))
            existing.add(c["source_id"])
            added_cases += 1
        terms = set((await s.execute(select(Glossary.term))).scalars().all())
        for g in glossary:
            if g["term"] in terms:
                continue
            s.add(Glossary(term=g["term"], definition=g["definition"], category=g.get("category", "general")))
            terms.add(g["term"])
            added_terms += 1

    # построение индексов и сохранение эмбеддингов в knowledge_base.embedding
    from backend.retrieval.index_manager import build_retriever

    all_cases, all_gl = await load_kb_from_db()
    retriever = await asyncio.to_thread(build_retriever, all_cases, all_gl)
    from backend.data.knowledge_base import contextualize
    from backend.retrieval.index_manager import case_payload

    emb = retriever.embedder
    async with session_scope() as s:
        rows = (await s.execute(select(KnowledgeBase))).scalars().all()
        by_kind: dict[str, list] = {}
        for r in rows:
            by_kind.setdefault(r.kind, []).append(r)
        for kind, rs in by_kind.items():
            dicts = [{"kind": r.kind, "series_features": r.series_features, "meta": r.meta, "best_model": r.best_model,
                      "problem_type": r.problem_type, "description": r.description} for r in rs]
            import numpy as np

            num = np.vstack([numeric_of(d) for d in dicts])
            texts = [contextualize(kind, d["description"], case_payload(d)) for d in dicts]
            vecs = emb.embed(kind, num, texts)
            for r, v in zip(rs, vecs, strict=True):
                r.embedding = [round(float(x), 5) for x in v]
    stats = {"added_cases": added_cases, "added_terms": added_terms, "total_cases": len(all_cases),
             "total_terms": len(all_gl), "index_docs": retriever.size}
    await dispose_engine()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="удалить сгенерированные записи перед импортом")
    a = ap.parse_args()
    print(asyncio.run(import_all(a.reset)))


if __name__ == "__main__":
    main()
