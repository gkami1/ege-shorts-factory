# One image for both the local CPU box and the GPU pod.
#   CPU:  docker build -t ege-factory .
#   GPU:  docker build -t ege-factory:gpu \
#           --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 \
#           --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 .
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    FACTORY_CACHE=/cache HF_HOME=/cache/hf TORCH_HOME=/cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential pkg-config git curl ca-certificates \
        libcairo2-dev libpango1.0-dev ffmpeg fontconfig fonts-liberation2 \
        texlive-xetex texlive-latex-extra texlive-lang-cyrillic texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*

# dvisvgm is packaged separately on 24.04 (Manim converts TeX output with it)
RUN apt-get update && apt-get install -y --no-install-recommends dvisvgm && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip && pip install torch --index-url ${TORCH_INDEX}

COPY tools/install_fonts.sh /tmp/install_fonts.sh
RUN pip install fonttools && bash /tmp/install_fonts.sh

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[align,dev]"
CMD ["bash"]
