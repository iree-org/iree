# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for running SHARK tank: https://github.com/nod-ai/SHARK.

# Ubuntu 22.04
FROM ubuntu@sha256:4b1d0c4a2d2aaf63b37111f34eb9fa89fa1bf53dd6e4ca954d47caebca4005c2

SHELL ["/bin/bash", "-e", "-u", "-o", "pipefail", "-c"]

######## Basic ########
WORKDIR /base

# Must set the timezone explicitly to avoid hanging when installing tzdata.
# https://grigorkh.medium.com/fix-tzdata-hangs-docker-image-build-cdb52cc3360d
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
  && apt-get install -y \
    git \
    wget \
    cmake \
    ninja-build \
    clang \
    lld

WORKDIR /
##############

######## Python ########
WORKDIR /install-python

ARG PYTHON_VERSION=3.11

COPY runtime/bindings/python/iree/runtime/build_requirements.txt build_tools/docker/context/install_python_deps.sh ./
RUN ./install_python_deps.sh "${PYTHON_VERSION}" \
  && rm -rf /install-python
WORKDIR /

ENV PYTHON_BIN /usr/bin/python3
##############

######## Cuda ########
WORKDIR /install-cuda

# We need CUDA Toolkit and CuDNN in order to run the Tensorflow XLA baselines.
ARG NVIDIA_TOOLKIT_DEB="cuda-repo-ubuntu2204-11-7-local_11.7.0-515.43.04-1_amd64.deb"
ARG NVIDIA_CUDNN_DEB="cudnn-local-repo-ubuntu2204-8.7.0.84_1.0-1_amd64.deb"

RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_TOOLKIT_DEB}"
RUN wget -q "https://storage.googleapis.com/iree-shared-files/${NVIDIA_CUDNN_DEB}"

# Install CUDA Toolkit. Instructions from https://developer.nvidia.com/cuda-downloads.
RUN dpkg -i "${NVIDIA_TOOLKIT_DEB}" \
  && cp /var/cuda-repo-ubuntu2204-11-7-local/cuda-46B62B5F-keyring.gpg /usr/share/keyrings/ \
  && apt-get update \
  && apt-get -y install cuda-toolkit-11.7

# Install CuDNN. Instructions from https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html.
RUN dpkg -i "${NVIDIA_CUDNN_DEB}" \
  && cp /var/cudnn-local-repo-ubuntu2204-8.7.0.84/cudnn-local-BF23AD8A-keyring.gpg /usr/share/keyrings/ \
  && apt-get update \
  && apt-get -y install libcudnn8 \
  && apt-get -y install libcudnn8-dev \
  && rm -rf /install-cuda

WORKDIR /
##############

######## Vulkan ########
WORKDIR /install-vulkan
ARG VULKAN_SDK_VERSION=1.2.154.0

RUN wget -q \
  # This file disappeared from the canonical source:
  # "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-${VULKAN_SDK_VERSION}.tar.gz"
  "https://storage.googleapis.com/iree-shared-files/vulkansdk-linux-${VULKAN_SDK_VERSION}.tar.gz" \
  && mkdir -p /opt/vulkan-sdk \
  && tar -xzf "vulkansdk-linux-${VULKAN_SDK_VERSION}.tar.gz" -C /opt/vulkan-sdk \
  && rm -rf /install-vulkan
WORKDIR /

ENV VULKAN_SDK="/opt/vulkan-sdk/${VULKAN_SDK_VERSION}/x86_64"
ENV PATH="${VULKAN_SDK}/bin:$PATH"
############## \
