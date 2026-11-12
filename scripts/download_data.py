"""Загрузка открытых датасетов.

python scripts/download_data.py            # UCI Student Performance (обязательно)
python scripts/download_data.py --kaggle   # + Kaggle через kagglehub (нужен kaggle.json)
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from backend.core.config import get_settings  # noqa: E402
from backend.tabular.datasets import UCI_URL  # noqa: E402

KAGGLE = ["larsen0966/student-performance-data-set", "anlgrbz/student-demographics-online-education-dataoulad"]


def download_uci(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "student-mat.csv").exists() and (dest / "student-por.csv").exists():
        print("UCI уже загружен:", dest)
        return
    r = httpx.get(UCI_URL, timeout=60, follow_redirects=True)
    r.raise_for_status()
    outer = zipfile.ZipFile(io.BytesIO(r.content))
    inner_name = next(n for n in outer.namelist() if n.endswith("student.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    for n in inner.namelist():
        if n.endswith(".csv") or n.endswith(".txt"):
            (dest / Path(n).name).write_bytes(inner.read(n))
    print("UCI Student Performance ->", dest)


def download_kaggle(dest: Path) -> None:  # pragma: no cover - требует учётку Kaggle
    try:
        import kagglehub
    except ImportError:
        print("pip install kagglehub")
        return
    for ds in KAGGLE:
        path = kagglehub.dataset_download(ds)
        print(ds, "->", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle", action="store_true")
    a = ap.parse_args()
    raw = get_settings().raw_dir
    download_uci(raw / "uci_student")
    if a.kaggle:
        download_kaggle(raw / "kaggle")
