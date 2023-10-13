# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# 20.04
FROM ubuntu@sha256:b795f8e0caaaacad9859a9a38fe1c78154f8301fdaf0872eaf1520d66d9c0b98

SHELL ["/bin/bash", "-e", "-u", "-o", "pipefail", "-c"]

# Disable apt-key parse waring. If someone knows how to do whatever the "proper"
# thing is then feel free. The warning complains about parsing apt-key output,
# which we're not even doing.
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

######## Basic stuff ########
WORKDIR /install-basics
COPY build_tools/docker/context/install_basics.sh ./
RUN ./install_basics.sh && rm -rf /install-basics

# Install the oldest supported compiler tools
ARG LLVM_VERSION=9
ENV CC /usr/bin/clang-${LLVM_VERSION}
ENV CXX /usr/bin/clang++-${LLVM_VERSION}

COPY build_tools/docker/context/install_iree_deps.sh ./
# We need DEBIAN_FRONTEND and TZ for the tzdata package needed by some IREE dependencies.
RUN  DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC ./install_iree_deps.sh "${LLVM_VERSION}" \
  && rm -rf /install-basics

######## ccache ########

WORKDIR /install-ccache

COPY build_tools/docker/context/install_ccache.sh ./
RUN ./install_ccache.sh && rm -rf /install-ccache

######## CMake ########
WORKDIR /install-cmake

# Install our minimum supported CMake version, which may be ahead of apt-get's version.
ENV CMAKE_VERSION="3.21.6"

COPY build_tools/docker/context/install_cmake.sh ./
RUN ./install_cmake.sh "${CMAKE_VERSION}" && rm -rf /install-cmake

##############

######## Bazel ########
WORKDIR /install-bazel
COPY build_tools/docker/context/install_bazel.sh .bazelversion ./
RUN ./install_bazel.sh && rm -rf /install-bazel

##############

######## Python ########

WORKDIR /install-python

# Minimum supported Python version
ARG PYTHON_VERSION=3.9

# Versions for things required to build IREE should match the minimum
# supported versions in the requirements file. There doesn't appear to be a
# pip-native way to get the minimum versions, but this hack works for simple
# files, at least.
COPY runtime/bindings/python/iree/runtime/build_requirements.txt build_tools/docker/context/install_python_deps.sh ./
RUN sed -i 's/>=/==/' build_requirements.txt \
  && ./install_python_deps.sh "${PYTHON_VERSION}" \
  && rm -rf /install-python

ENV PYTHON_BIN /usr/bin/python3

##############

######## IREE ROCM DEPS ########
WORKDIR /install-rocm
ARG ROCM_VERSION=5.2.1
ARG AMDGPU_VERSION=22.20.1
COPY build_tools/docker/context/install_rocm.sh ./
RUN ./install_rocm.sh "${ROCM_VERSION}" "${AMDGPU_VERSION}" && rm -rf /install-rocm
WORKDIR /
##############

######## IREE CUDA DEPS ########
ENV IREE_CUDA_DEPS_DIR="/usr/local/iree_cuda_deps"
COPY build_tools/docker/context/fetch_cuda_deps.sh /usr/local/bin
RUN /usr/local/bin/fetch_cuda_deps.sh "${IREE_CUDA_DEPS_DIR}"
##############

######## Vulkan ########
WORKDIR /install-vulkan
ARG VULKAN_SDK_VERSION=1.3.250
COPY build_tools/docker/context/install_vulkan.sh ./
RUN ./install_vulkan.sh "${VULKAN_SDK_VERSION}" && rm -rf /install-vulkan
WORKDIR /
##############

### Clean up
RUN apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /
