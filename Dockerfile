FROM python:3.11-slim

WORKDIR /usr/src/app

COPY pyproject.toml .
COPY MANIFEST.in .
COPY src/nesse ./nesse

RUN \
    apt-get update && \
    apt-get upgrade -y && \
    pip install --upgrade pip && \
    apt-get clean && \
    apt-get update && \
    apt-get install -y build-essential gcc clang clang-tools cmake cppcheck valgrind afl gcc-multilib && \
    rm -rf /var/lib/apt/lists/* &&\
    pip install . \