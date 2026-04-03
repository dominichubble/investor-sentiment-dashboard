# Smaller runtime for Render free tier: pip install requirements-lean.txt only.
# Set env LEAN_API=1 on the service (see .env.example).
# Build from repo with context = backend (same as main Dockerfile).

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-lean.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

ENTRYPOINT ["/docker-entrypoint.sh"]
