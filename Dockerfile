FROM python:3.11-slim AS compile

WORKDIR /usr/src/app

RUN \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

COPY pyproject.toml .
COPY MANIFEST.in .
COPY setup.py .
COPY src/nesse ./src/nesse

RUN pip install .

FROM python:3.11-slim AS build

COPY --from=compile /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
#CMD ["python"]