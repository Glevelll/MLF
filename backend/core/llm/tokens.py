"""Подсчёт токенов, когда провайдер их не возвращает (tiktoken, иначе эвристика)."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # нет сети для загрузки словаря / нет пакета
        return None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:  # pragma: no cover
            pass
    # грубая оценка: ~4 символа на токен для латиницы, ~2.5 для кириллицы
    cyr = sum(1 for ch in text if "Ѐ" <= ch <= "ӿ")
    return max(1, int(cyr / 2.5 + (len(text) - cyr) / 4))
