# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for running mmperf: https://github.com/mmperf/mmperf.
#
# mmperf benchmarks matrix-multiplication workloads on IREE and various backends
# such as OpenBLAS, MKL, TVM, Halide, CuBLAS, etc.
#
# These backends are included either in this image or as a submodule in the
# mmperf repo. Later versions of Clang, LLVM, Python and Ubuntu are needed
# to satisfy the dependency requirements of the backends.

# Ubuntu 22.04.
FROM ubuntu@sha256:817cfe4672284dcbfee885b1a66094fd907630d610cab329114d036716be49ba

######## Basic ########
RUN apt-get update \
  && apt-get install -y \
    git \
    wget \
    cmake \
    curl \
    ninja-build
##############

######## Clang/LLVM ########
RUN apt-get update \
  && apt-get install -y \
    llvm-14 \
    llvm-14-dev \
    clang-14 \
    clang-tools-14 \
    libclang-common-14-dev \
    libclang-14-dev \
    libclang1-14 \
    clang-format-14 \
    clangd-14 \
    clang-tidy-14 \
    lldb-14 \
    lld-14 \
    libmlir-14-dev \
    mlir-14-tools \
  && ln -s /usr/lib/llvm-14/bin/clang /usr/bin/clang \
  && ln -s /usr/lib/llvm-14/bin/clang++ /usr/bin/clang++
##############

######## Python ########
WORKDIR /install-python

ARG PYTHON_VERSION=3.10

COPY runtime/bindings/python/iree/runtime/build_requirements.txt build_tools/docker/context/install_python_deps.sh ./
RUN ./install_python_deps.sh "${PYTHON_VERSION}" \
  && rm -rf /install-python

ENV PYTHON_BIN /usr/bin/python3
##############

######## CUDA ########
RUN apt-get update \
  && apt-get install -y \
    nvidia-cuda-toolkit \
  && mkdir -p "/usr/nvvm/libdevice" \
  && ln -s "/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc" "/usr/nvvm/libdevice/libdevice.10.bc"
##############

######## MKL ########
WORKDIR /install-mkl

RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
    && apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
    && sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list' \
    && apt-get update \
    && apt-get install -y intel-mkl-64bit-2018.2-046 \
    && rm -rf /install-mkl

WORKDIR /

ENV MKL_DIR="/opt/intel/mkl"
##############

######## OPENBLAS ########
RUN apt-get update \
  && apt-get install -y libopenblas-dev
##############

######## BLIS ########
WORKDIR /install-blis

RUN git clone --recurse-submodules https://github.com/flame/blis \
  && cd blis \
  && ./configure --prefix=/opt/blis --enable-cblas -c amd64 \
  && make -j 32 \
  && make install \
  && rm -rf /install-blis

WORKDIR /

ENV BLIS_DIR="/opt/blis"
##############

######## MMPERF ########
COPY build_tools/docker/context/setup_mmperf.sh /usr/local/bin

ARG MMPERF_SHA="ae523a31449a58ef39592c6b8cf9c042e0a55a1f"

# Generate a version of mmperf for CPU.
RUN mkdir -p "/usr/local/src/mmperf" \
    && /usr/local/bin/setup_mmperf.sh "/usr/local/src/mmperf" "${MMPERF_SHA}"

ENV MMPERF_REPO_DIR="/usr/local/src/mmperf/mmperf"
############## \
