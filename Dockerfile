# Stage 1: Build
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

RUN groupadd -r app && useradd -r -g app -d /app -s /bin/false app

COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache /root/.cache

COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1 SMARTHUB_ALLOWED_EXTRACT_DIR=/app

EXPOSE 8003

USER app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
