FROM python:3.10-slim

WORKDIR /app

# Build tools for TenSEAL (CKKS homomorphic encryption library)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    protobuf-compiler \
    libprotobuf-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Render uses PORT env variable
CMD gunicorn --bind 0.0.0.0:$PORT --timeout 300 --workers 1 server_fhe:app
