# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# 18.04
FROM ubuntu@sha256:fd25e706f3dea2a5ff705dbc3353cf37f08307798f3e360a13e9385840f73fb3

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
    gnupg2 \
    lsb-release

# Install the oldest supported compiler tools
ARG LLVM_VERSION=9
ENV CC /usr/bin/clang-${LLVM_VERSION}
ENV CXX /usr/bin/clang++-${LLVM_VERSION}

COPY build_tools/docker/context/install_iree_deps.sh ./
RUN ./install_iree_deps.sh "${LLVM_VERSION}" \
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
ARG PYTHON_VERSION=3.7

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

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl libnuma-dev gnupg \
  && curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
  && printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION/ ubuntu main" | tee /etc/apt/sources.list.d/rocm.list \
  && printf "deb [arch=amd64] https://repo.radeon.com/amdgpu/$AMDGPU_VERSION/ubuntu bionic main" | tee /etc/apt/sources.list.d/amdgpu.list \
  && apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  libelf1 \
  kmod \
  file \
  rocm-dev \
  build-essential
WORKDIR /
##############

######## IREE CUDA DEPS ########
ENV IREE_CUDA_DEPS_DIR="/usr/local/iree_cuda_deps"
COPY build_tools/docker/context/fetch_cuda_deps.sh /usr/local/bin
RUN /usr/local/bin/fetch_cuda_deps.sh "${IREE_CUDA_DEPS_DIR}"
##############

######## Vulkan ########
WORKDIR /install-vulkan
ARG VULKAN_SDK_VERSION=1.2.154.0

RUN curl --silent --fail --show-error --location \
  # This file disappeared from the canonical source:
  # "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-${VULKAN_SDK_VERSION}.tar.gz"
  "https://storage.googleapis.com/iree-shared-files/vulkansdk-linux-${VULKAN_SDK_VERSION}.tar.gz" \
  --output vulkansdk.tar.gz \
  && mkdir -p /opt/vulkan-sdk \
  && tar -xzf vulkansdk.tar.gz -C /opt/vulkan-sdk \
  && rm -rf /install-vulkan
WORKDIR /

ENV VULKAN_SDK="/opt/vulkan-sdk/${VULKAN_SDK_VERSION}/x86_64"

ENV PATH="${VULKAN_SDK}/bin:$PATH"

# Symlink the Vulkan loader to a system library directory. This is needed to
# allow Vulkan applications to find the Vulkan loader. It also avoids using
# LD_LIBRARY_PATH, which is not supported well by Docker.
RUN ln -s "${VULKAN_SDK}/lib/libvulkan.so" /usr/lib/x86_64-linux-gnu/ \
  && ln -s "${VULKAN_SDK}/lib/libvulkan.so.1" /usr/lib/x86_64-linux-gnu/

##############


######## GCC ########
WORKDIR /

# Avoid apt-add-repository, which requires software-properties-common, which is
# a rabbit hole of python version compatibility issues. See
# https://mondwan.blogspot.com/2018/05/alternative-for-add-apt-repository-for.html
# We use gcc-9 because it's what manylinux had (at time of authorship) and
# we don't aim to support older versions. We need a more modern lld to handle
# --push-state flags
RUN echo "deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic main" >> /etc/apt/sources.list  \
  && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1E9377A2BA9EF27F \
  && apt-get update \
  && apt-get install -y gcc-9 g++-9

##############

### Clean up
RUN apt-get clean \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /
