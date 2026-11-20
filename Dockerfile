FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY backend ./backend
COPY frontend ./frontend
RUN pip install --upgrade pip && pip install ".[ui]"
COPY scripts ./scripts
COPY run.py alembic.ini ./
COPY data/kb ./data/kb
COPY data/raw/uci_student ./data/raw/uci_student
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000 7860
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]
