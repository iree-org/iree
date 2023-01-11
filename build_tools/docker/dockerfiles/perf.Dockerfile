# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image that includes basic packages, compiler and Python support for
# running perf-related workflows e.g. mmperf, convperf.

# Ubuntu 22.04.
FROM ubuntu@sha256:817cfe4672284dcbfee885b1a66094fd907630d610cab329114d036716be49ba

######## Basic ########
RUN apt-get update \
  && apt-get install -y \
    git \
    wget \
    cmake \
    curl \
    ninja-build \
    numactl
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
