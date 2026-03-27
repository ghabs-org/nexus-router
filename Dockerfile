FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and default policies
COPY src/ ./src/
COPY policies/ ./policies/

# Runtime volumes — mounted by docker-compose:
#   /app/catalog  → model catalog (raw + normalized)
#   /app/state    → runtime provider health
#   /app/data     → SQLite routing history
VOLUME ["/app/catalog", "/app/state", "/app/data"]

EXPOSE 7771

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7771/health', timeout=3)" || exit 1

CMD ["python", "-m", "src.server", "--port", "7771"]
