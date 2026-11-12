"""Конфигурация приложения (pydantic-settings, переменные окружения с префиксом MLF_)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MLF_", extra="ignore")

    app_env: Literal["local", "prod", "test"] = "local"
    data_dir: Path = PROJECT_ROOT / "data"
    database_url: str | None = None

    # --- Безопасность ---
    jwt_secret: str = "local-dev-secret-change-me-in-production-0123456789"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 720
    admin_email: str | None = None  # пользователь с этим email при регистрации получает роль admin

    # --- LLM ---
    llm_provider: Literal["mock", "ollama", "hf", "openai", "anthropic"] = "mock"
    llm_fallback_providers: list[str] = Field(default_factory=lambda: ["mock"])
    llm_model: str = "qwen2.5:7b"
    ollama_url: str = "http://localhost:11434"
    hf_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-haiku-5-5"
    llm_timeout: float = 120.0
    llm_max_retries: int = 2
    llm_cache_enabled: bool = True

    # --- Инфраструктура ---
    task_backend: Literal["inline", "celery"] = "inline"
    redis_url: str = "redis://localhost:6379/0"
    cache_backend: Literal["db", "redis"] = "db"
    vector_backend: Literal["memory", "qdrant"] = "memory"
    qdrant_url: str = "http://localhost:6333"
    text_embedder: Literal["lsa", "sentence_transformers"] = "lsa"

    # --- Агенты ---
    max_iterations: int = 3
    critic_threshold: float = 6.0
    high_risk_threshold: float = 0.7
    min_auc: float = 0.7
    min_r2: float = 0.3
    rag_top_k: int = 5
    random_seed: int = 42

    log_level: str = "INFO"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def kb_dir(self) -> Path:
        return self.data_dir / "kb"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite+aiosqlite:///{(self.data_dir / 'mlf.db').as_posix()}"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.artifacts_dir, self.index_dir, self.kb_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
