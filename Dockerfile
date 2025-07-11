FROM python:3.11-slim

WORKDIR /usr/src/app

RUN \
    apt-get update && \
    apt-get upgrade -y && \
    pip install --upgrade pip && \
    apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY MANIFEST.in .
COPY setup.py .
COPY src/nesse ./src/nesse

RUN pip install .