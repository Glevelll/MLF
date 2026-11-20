"""Источники данных: UCI Student Performance и синтетический генератор в стиле OULAD."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.core.config import get_settings
from backend.tabular.preprocessing import STUDENT_GROUP

UCI_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"

UCI_NUMERIC = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout",
               "Dalc", "Walc", "health", "absences", "G1", "G2", "G3"]
UCI_MATCH_COLUMNS = ("school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
                     "Mjob", "Fjob", "reason", "nursery", "internet")


def _uci_group(values: tuple[object, ...]) -> str:
    # The matching columns follow the merge script shipped with the UCI dataset.
    profile = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(profile.encode()).hexdigest()


def load_uci(subject: str = "both", raw_dir: Path | None = None) -> pd.DataFrame:
    """Датасет Cortez & Silva (2008): 33 признака, G3 — итоговая оценка 0..20."""
    raw_dir = raw_dir or get_settings().raw_dir / "uci_student"
    parts = []
    for subj in (["mat", "por"] if subject == "both" else [subject]):
        path = raw_dir / f"student-{subj}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} не найден — запустите scripts/download_data.py")
        df = pd.read_csv(path, sep=";")
        df.insert(0, "subject", subj)
        df[STUDENT_GROUP] = [_uci_group(row) for row in df[list(UCI_MATCH_COLUMNS)].itertuples(index=False, name=None)]
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    for c in UCI_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "student_id", [f"uci_{i:05d}" for i in range(len(df))])
    return df


REGIONS = ["North", "South", "East", "West", "Central", "Scotland", "Wales"]
SPECIALTIES = ["CS", "Math", "Economics", "Biology", "History", "Law", "Engineering"]
EDUCATION = ["No formal", "Lower than A", "A level", "HE qualification", "Post grad"]
IMD = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]


def generate_synthetic(n: int = 1000, seed: int = 0, noise: float = 1.0, dropout_shift: float = 0.0,
                       missing_rate: float = 0.03, n_noise_features: int = 0, nonlinear: bool = True) -> pd.DataFrame:
    """Синтетические траектории обучения (OULAD-подобная схема).

    Целевые переменные: final_score (0..100, регрессия) и dropout (0/1, классификация).
    Скрытая «вовлечённость» студента формирует посещаемость, сдачу ДЗ, баллы тестов,
    время в LMS и активность на форуме; итог зависит от них нелинейно + шум.
    """
    rng = np.random.default_rng(seed)
    engagement = rng.normal(0, 1, n)
    ability = rng.normal(0, 1, n)
    gender = rng.choice(["F", "M"], n)
    age_band = rng.choice(["0-25", "25-35", "35-55", "55+"], n, p=[0.55, 0.28, 0.15, 0.02])
    region = rng.choice(REGIONS, n)
    education = rng.choice(EDUCATION, n, p=[0.05, 0.3, 0.4, 0.2, 0.05])
    edu_level = np.array([EDUCATION.index(e) for e in education])
    imd = rng.choice(IMD, n)
    imd_level = np.array([IMD.index(e) for e in imd])
    specialty = rng.choice(SPECIALTIES, n)
    disability = rng.choice(["N", "Y"], n, p=[0.9, 0.1])
    scholarship = rng.choice(["no", "yes"], n, p=[0.7, 0.3])
    prev_attempts = rng.poisson(np.clip(0.3 - 0.2 * ability, 0.02, None))
    credits = rng.choice([30, 60, 90, 120], n, p=[0.2, 0.5, 0.2, 0.1])
    workload = (credits - 60) / 30

    attendance = 1 / (1 + np.exp(-(1.2 + 1.1 * engagement - 0.3 * workload + rng.normal(0, 0.4, n))))
    hw_ratio = 1 / (1 + np.exp(-(1.0 + 1.3 * engagement + 0.3 * ability + rng.normal(0, 0.5, n))))
    lms_hours = np.clip(rng.gamma(2 + 1.5 * np.clip(engagement + 1.5, 0.1, None), 6), 0, None)
    forum_posts = rng.poisson(np.clip(2 + 3 * engagement + 1.5 * (specialty == "CS"), 0.1, None))
    late = rng.poisson(np.clip(2 - 1.2 * engagement + 0.5 * workload, 0.05, None))
    base = 60 + 9 * ability + 6 * engagement + 2 * edu_level + 1.5 * imd_level - 4 * prev_attempts
    test1 = np.clip(base + rng.normal(0, 10 * noise, n), 0, 100)
    trend = rng.normal(2 * engagement, 3, n)
    test2 = np.clip(base + trend + rng.normal(0, 10 * noise, n), 0, 100)
    test3 = np.clip(base + 2 * trend + rng.normal(0, 10 * noise, n), 0, 100)
    hw_score = np.clip(55 + 12 * hw_ratio + 6 * ability + rng.normal(0, 8 * noise, n), 0, 100)

    latent = (0.25 * test1 + 0.3 * test2 + 0.35 * test3 + 12 * attendance + 8 * hw_ratio
              + 0.08 * lms_hours - 1.5 * late)
    if nonlinear:
        latent += -10 * (attendance < 0.5) + 4 * np.tanh((forum_posts - 3) / 3) - 3 * (disability == "Y") * (attendance < 0.7)
    final = np.clip(latent - 15 + rng.normal(0, 6 * noise, n), 0, 100)
    drop_logit = (-2.4 + dropout_shift - 2.2 * (attendance - 0.75) * 3 - 0.05 * (test2 - 60)
                  + 0.35 * late + 0.6 * prev_attempts - 0.4 * (scholarship == "yes") + rng.normal(0, 0.6 * noise, n))
    dropout = (rng.random(n) < 1 / (1 + np.exp(-drop_logit))).astype(int)

    df = pd.DataFrame({
        "student_id": [f"syn{seed}_{i:05d}" for i in range(n)],
        "gender": gender, "age_band": age_band, "region": region, "highest_education": education,
        "imd_band": imd, "disability": disability, "specialty": specialty, "scholarship": scholarship,
        "num_prev_attempts": prev_attempts, "studied_credits": credits,
        "attendance_rate": attendance.round(3), "hw_submitted_ratio": hw_ratio.round(3),
        "hw_avg_score": hw_score.round(1), "test1_score": test1.round(1), "test2_score": test2.round(1),
        "test3_score": test3.round(1), "lms_hours": lms_hours.round(1), "forum_posts": forum_posts,
        "late_submissions": late, "final_score": final.round(1), "dropout": dropout,
    })
    for j in range(n_noise_features):
        df[f"noise_{j}"] = rng.normal(0, 1, n).round(3)
    if missing_rate > 0:
        for c in ["imd_band", "hw_avg_score", "lms_hours", "forum_posts", "test1_score", "attendance_rate"]:
            mask = rng.random(n) < missing_rate
            df.loc[mask, c] = np.nan
    return df


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        head = f.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    return pd.read_csv(path, sep=sep)
