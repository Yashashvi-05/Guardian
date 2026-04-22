FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps first for layer cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY guardian/ ./guardian/
COPY guardian-mcp-proxy/ ./guardian-mcp-proxy/
COPY setup.py .
RUN pip install --no-cache-dir -e . --no-deps
RUN pip install --no-cache-dir -e ./guardian-mcp-proxy

# Data directories
RUN mkdir -p guardian/data guardian/checkpoints

# Environment variable validation (non-fatal if unset)
ENV GUARDIAN_PORT=7860
ENV MCP_UPSTREAM_URL=""
ENV HMAC_SECRET="changeme"
ENV LOG_LEVEL="info"

EXPOSE 7860 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl -f http://localhost:${GUARDIAN_PORT}/ || exit 1

CMD ["python", "-m", "guardian.dashboard.app"]
