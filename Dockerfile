FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KNOWLEDGEPILOT_HOST=0.0.0.0 \
    KNOWLEDGEPILOT_PORT=7860 \
    KNOWLEDGEPILOT_GENERATOR_PROVIDER=ollama \
    KNOWLEDGEPILOT_OLLAMA_BASE_URL=https://www.google.com/search?q=http://127.0.0.1:11434 \
    KNOWLEDGEPILOT_OLLAMA_MODEL=llama3.2:1b \
    KNOWLEDGEPILOT_STORAGE_PATH=/tmp/kp_data/index_state.json \
    HF_HOME=/tmp/huggingface \
    OLLAMA_MODELS=/tmp/ollama_models

WORKDIR /app

# Install curl + zstd for Ollama and clean up apt cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates zstd && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama inside the container
RUN curl -fsSL https://ollama.com/install.sh | sh

COPY pyproject.toml README.md ./
COPY src ./src
COPY start.sh ./start.sh

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . && \
    chmod +x /app/start.sh && \
    mkdir -p /tmp/kp_data /tmp/huggingface /tmp/ollama_models && \
    chmod -R 777 /tmp/kp_data /tmp/huggingface /tmp/ollama_models

EXPOSE 7860

CMD ["/app/start.sh"]
