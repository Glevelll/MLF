"""Генерация справочника (≥500 терминов) в data/kb/glossary.jsonl.

python scripts/generate_glossary.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from glossary_data import AGENTIC, CORE, EXTRA, FACTORS, ISSUES, METRICS, MODELS, SYN_ATTRS, TESTS, UCI_ATTRS  # noqa: E402

from backend.core.config import get_settings  # noqa: E402
from backend.core.llm.tools import default_registry  # noqa: E402
from backend.tabular.features import AVAILABLE_OPS  # noqa: E402


def build() -> list[dict]:
    items: list[dict] = []

    def add(term: str, definition: str, category: str) -> None:
        items.append({"term": term, "definition": definition, "category": category})

    for t, d, c in CORE + EXTRA:
        add(t, d, c)
    for name, (defn, formula, rng, pitfall, edu) in METRICS.items():
        add(name, f"Метрика: {defn}. Диапазон {rng}.", "metrics")
        add(f"{name}: формула", f"{name} = {formula}.", "metrics")
        add(f"{name}: подводные камни", f"{name}: {pitfall}.", "metrics")
        add(f"{name}: применение в образовательной аналитике", f"{name}: {edu}.", "metrics")
    for name, (defn, when, pros, cons, hp) in MODELS.items():
        add(name, f"Модель: {defn}.", "models")
        add(f"{name}: когда применять", f"{name} подходит, когда: {when}.", "models")
        add(f"{name}: преимущества", f"{name}: {pros}.", "models")
        add(f"{name}: ограничения", f"{name}: {cons}.", "models")
        add(f"{name}: ключевые гиперпараметры", f"Настраивают: {hp}.", "models")
    for name, desc, action in FACTORS:
        add(f"Фактор: {name}", f"{name} — {desc}.", "education")
        add(f"Вмешательство: {name}", f"При неблагоприятном значении фактора «{name}»: {action}.", "education")
    for name, (defn, assump, interp) in TESTS.items():
        add(name, f"Статистический тест: {defn}.", "statistics")
        add(f"{name}: допущения", f"{name}: {assump}.", "statistics")
        add(f"{name}: интерпретация", f"{name}: {interp}.", "statistics")
    for t, d in AGENTIC:
        add(t, d + ".", "agents")
    reg = default_registry()
    for tool in reg.list():
        add(f"Инструмент {tool.name}", tool.description, "tools")
        for field, info in tool.args_model.model_fields.items():
            desc = info.description or "параметр инструмента"
            default = info.default if not callable(info.default) else None
            add(f"{tool.name}.{field}", f"Аргумент инструмента {tool.name}: {desc}; значение по умолчанию {default!r}.",
                "hyperparameters")
    for op, desc in AVAILABLE_OPS.items():
        add(f"Операция feature engineering: {op}", desc + ".", "features")
    for col, desc in UCI_ATTRS.items():
        add(f"UCI: {col}", f"Признак набора UCI Student Performance: {desc}.", "data")
    for col, desc in SYN_ATTRS.items():
        add(f"OULAD-схема: {col}", f"Признак синтетического OULAD-подобного набора: {desc}.", "data")
    for code, desc in ISSUES.items():
        add(f"Замечание Validator: {code}", desc + ".", "validation")
    # уникальность терминов
    seen, uniq = set(), []
    for it in items:
        if it["term"] not in seen:
            seen.add(it["term"])
            uniq.append(it)
    return uniq


def main() -> None:
    items = build()
    out = get_settings().kb_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / "glossary.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"glossary: {len(items)} терминов -> {path}")


if __name__ == "__main__":
    main()
