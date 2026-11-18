FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KNOWLEDGEPILOT_HOST=0.0.0.0 \
    KNOWLEDGEPILOT_PORT=8000 \
    KNOWLEDGEPILOT_STORAGE_PATH=/app/data/index_state.json

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "-m", "knowledgepilot.app", "--serve"]
