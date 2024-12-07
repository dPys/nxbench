FROM rapidsai/base:24.10-cuda12.5-py3.12

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PATH=/opt/conda/bin:${PATH}

WORKDIR /app

COPY . /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    g++ \
    git \
    curl \
    vim \
    python3-dev \
    python3-pip \
    && conda install -c rapidsai -c conda-forge -y cugraph \
    && apt-get clean \
    && apt-get purge -y \
    && rm -rf /var/lib/{apt,dpkg,cache,log}/ \
    && rm -rf /tmp/* /var/tmp/* \
    && conda clean --all --yes

RUN pip install --no-cache-dir \
    nxbench \
    nx-parallel \
    python-graphblas \
    ipython \
    && mkdir -p ~/.nxbench/data && rm -rf /root/.cache/pip

ENV NXBENCH_HOME=/root/.nxbench
ENV NXBENCH_CONFIG_FILE=/app/configs/example.yaml

CMD ["nxbench"]

LABEL maintainer="Derek Alexander <dpysalexander@example.com>"
LABEL description="Container image for nxbench graph algorithm benchmarking suite"
LABEL version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/dpys/nxbench"

# USER rapids
