"""Спецификации наборов данных для базы знаний и оценки (непересекающиеся seed-ы по split)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backend.tabular.datasets import generate_synthetic, load_uci

SPLIT_OFFSET = {"kb": 0, "eval": 100_003}


@dataclass
class DatasetSpec:
    id: str
    source: str  # uci_mat | uci_por | uci_both | synthetic
    target: str
    problem_type: str
    binarize_threshold: float | None = None
    seed: int = 0
    n: int = 1000
    noise: float = 1.0
    dropout_shift: float = 0.0
    missing_rate: float = 0.03
    n_noise: int = 0
    nonlinear: bool = True
    subsample: float = 1.0
    drop_cols: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_specs(n: int, split: str = "kb", seed: int = 0) -> list[DatasetSpec]:
    rng = np.random.default_rng(seed + SPLIT_OFFSET[split])
    specs = []
    for i in range(n):
        s = int(rng.integers(0, 2**31 - 1))
        if rng.random() < 0.4:
            subj = str(rng.choice(["mat", "por", "both"]))
            variant = rng.random()
            drop = []
            r = rng.random()
            if r < 0.45:
                drop = ["G1", "G2"]
            elif r < 0.65:
                drop = ["G2"]
            if variant < 0.45:
                spec = DatasetSpec(f"{split}_{i:04d}", f"uci_{subj}", "G3", "classification", 10.0)
            elif variant < 0.6:
                spec = DatasetSpec(f"{split}_{i:04d}", f"uci_{subj}", "G3", "classification", 12.0)
            else:
                spec = DatasetSpec(f"{split}_{i:04d}", f"uci_{subj}", "G3", "regression")
            spec.seed, spec.subsample, spec.drop_cols = s, float(rng.uniform(0.35, 1.0)), drop
        else:
            variant = rng.random()
            if variant < 0.5:
                spec = DatasetSpec(f"{split}_{i:04d}", "synthetic", "dropout", "classification")
            elif variant < 0.7:
                spec = DatasetSpec(f"{split}_{i:04d}", "synthetic", "final_score", "classification", 40.0)
            else:
                spec = DatasetSpec(f"{split}_{i:04d}", "synthetic", "final_score", "regression")
            spec.seed = s
            spec.n = int(np.exp(rng.uniform(np.log(200), np.log(4000))))
            spec.noise = float(rng.uniform(0.5, 2.0))
            spec.dropout_shift = float(rng.uniform(-1.2, 1.2))
            spec.missing_rate = float(rng.choice([0.0, 0.02, 0.05, 0.15]))
            spec.n_noise = int(rng.choice([0, 0, 5, 15]))
            spec.nonlinear = bool(rng.random() < 0.7)
            if rng.random() < 0.3:
                spec.drop_cols = ["test2_score", "test3_score"]
        specs.append(spec)
    return specs


def build_dataset(spec: DatasetSpec, uci_cache: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    if spec.source.startswith("uci"):
        subj = spec.source.split("_", 1)[1]
        cache = uci_cache if uci_cache is not None else {}
        if subj not in cache:
            cache[subj] = load_uci(subj)
        df = cache[subj]
        if spec.subsample < 1.0:
            df = df.sample(frac=spec.subsample, random_state=spec.seed % (2**31 - 1))
        df = df.reset_index(drop=True)
    else:
        df = generate_synthetic(spec.n, spec.seed, spec.noise, spec.dropout_shift, spec.missing_rate,
                                spec.n_noise, spec.nonlinear)
        other = "final_score" if spec.target == "dropout" else "dropout"
        df = df.drop(columns=[other])
    return df.drop(columns=[c for c in spec.drop_cols if c in df.columns])


def prepare_kwargs(spec: DatasetSpec) -> dict[str, Any]:
    return {"target": spec.target, "problem_type": spec.problem_type, "binarize_threshold": spec.binarize_threshold}
