# Ubuntu 22.04 (Jammy)
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV HOME=/root
WORKDIR /root

# OS deps + Python 3.11 + libs that fixed your .so errors + build tools + pkg-config
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        locales ca-certificates wget git curl \
        build-essential gfortran cmake pkg-config \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        zlib1g-dev \
        libblas-dev liblapack-dev libopenblas-dev \
        libhdf5-103 libhdf5-dev libsz2 \
        libblosc1 \
        libfftw3-dev \
        libgmp-dev libmpfr-dev \
        libxml2-dev libxslt1-dev \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# Install FEniCS (same script you used)
# (This drops Python packages under /usr/local/lib/python3.11/dist-packages)
RUN wget -q "https://fem-on-colab.github.io/releases/fenics-install-release-real.sh" -O /tmp/fenics-install.sh && \
    bash /tmp/fenics-install.sh && \
    rm /tmp/fenics-install.sh

# Install missing shared library dependencies for FEniCS
RUN apt-get update && apt-get install -y \
    libx11-6 libxext6 libxt6 libsm6 libice6 \
    libevent-2.1-7 libevent-core-2.1-7 libevent-pthreads-2.1-7 \
 && rm -rf /var/lib/apt/lists/*


# Python 3.11 scientific stack + Jupyter
# (Use python3.11 explicitly so these land in the same interpreter where FEniCS was installed)
RUN /usr/bin/python3.11 -m pip install --no-cache-dir \
      numpy scipy sympy mpmath jupyterlab ipykernel

# Jupyter defaults
EXPOSE 8888
CMD ["/usr/bin/python3.11", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
