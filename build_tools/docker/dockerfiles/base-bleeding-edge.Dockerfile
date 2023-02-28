# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A Docker image that contains all the latest stuff. We mostly test our oldest
# supported versions, but it's good to test the cutting edge also.

# 22.04
FROM ubuntu@sha256:4b1d0c4a2d2aaf63b37111f34eb9fa89fa1bf53dd6e4ca954d47caebca4005c2

SHELL ["/bin/bash", "-e", "-u", "-o", "pipefail", "-c"]

# Disable apt-key parse waring. If someone knows how to do whatever the "proper"
# thing is then feel free. The warning complains about parsing apt-key output,
# which we're not even doing.
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

######## Basic stuff ########
WORKDIR /install-basics
# Useful utilities for building child images. Best practices would tell us to
# use multi-stage builds
# (https://docs.docker.com/develop/develop-images/multistage-build/) but it
# turns out that Dockerfile is a thoroughly non-composable awful format and that
# doesn't actually work that well. These deps are pretty small.
RUN apt-get update \
  && apt-get install -y \
    git \
    unzip \
    wget \
    curl \
    gnupg2

# Install the latest supported compiler tools
ARG LLVM_VERSION=16
ENV CC /usr/bin/clang-${LLVM_VERSION}
ENV CXX /usr/bin/clang++-${LLVM_VERSION}

COPY build_tools/docker/context/install_iree_deps.sh ./
RUN echo "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy main" >> /etc/apt/sources.list \
  && curl https://apt.llvm.org/llvm-snapshot.gpg.key \
      | gpg --dearmor > /etc/apt/trusted.gpg.d/llvm-snapshot.gpg \
  && ./install_iree_deps.sh "${LLVM_VERSION}" \
  && rm -rf /install-basics

######## ccache ########

WORKDIR /install-ccache

COPY build_tools/docker/context/install_ccache.sh ./
RUN ./install_ccache.sh && rm -rf /install-ccache

######## CMake ########
WORKDIR /install-cmake

# Install the latest CMake version we support
ENV CMAKE_VERSION="3.24.3"

COPY build_tools/docker/context/install_cmake.sh ./
RUN ./install_cmake.sh "${CMAKE_VERSION}" && rm -rf /install-cmake

##############

######## Bazel ########
WORKDIR /install-bazel
COPY build_tools/docker/context/install_bazel.sh .bazelversion ./
RUN ./install_bazel.sh && rm -rf /install-bazel

##############

##############

WORKDIR /install-vulkan
RUN apt-get update \
  && apt-get install -y \
    # Modern Vulkan versions now available via apt
    libvulkan-dev \
    vulkan-tools

##############

######## Python ########
WORKDIR /install-python

# Max supported Python version
ARG PYTHON_VERSION=3.10

# Versions for things required to build IREE. We install the latest version of
# packages in the requirements file. Unlike most places in the project, these
# dependencies are *not* pinned in general. This adds non-determinism to the
# Docker image build, but reduces maintenance burden. If a new build fails
# because of a new package version, we should add a max version constraint to
# the requirement.
COPY runtime/bindings/python/iree/runtime/build_requirements.txt build_tools/docker/context/install_python_deps.sh ./
RUN ./install_python_deps.sh "${PYTHON_VERSION}" \
  && rm -rf /install-python

ENV PYTHON_BIN /usr/bin/python3

##############

######## IREE CUDA DEPS ########
ENV IREE_CUDA_DEPS_DIR="/usr/local/iree_cuda_deps"
COPY build_tools/docker/context/fetch_cuda_deps.sh /usr/local/bin
RUN /usr/local/bin/fetch_cuda_deps.sh "${IREE_CUDA_DEPS_DIR}"
##############

### Clean up
RUN apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /
