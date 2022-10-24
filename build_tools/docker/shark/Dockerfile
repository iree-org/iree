# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for running SHARK tank: https://github.com/nod-ai/SHARK.

FROM ubuntu@sha256:fd25e706f3dea2a5ff705dbc3353cf37f08307798f3e360a13e9385840f73fb3

######## Basic ########
WORKDIR /base

# Must set the timezone explicitly to avoid hanging when installing tzdata.
# https://grigorkh.medium.com/fix-tzdata-hangs-docker-image-build-cdb52cc3360d
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
  && apt-get install -y \
    git \
    wget

WORKDIR /
##############

######## Python ########
WORKDIR /install-python

RUN apt-get update \
  && apt-get install -y software-properties-common \
  && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt-get update \
  && apt-get install -y \
    python3.10 \
    python3.10-dev \
  && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
  && apt-get install -y \
    python3-pip \
    python3-setuptools \
    python3-distutils \
    python3-venv \
    python3.10-venv

ENV PYTHON_BIN /usr/bin/python3

WORKDIR /
##############

######## Vulkan ########
WORKDIR /install-vulkan
ARG VULKAN_SDK_VERSION=1.2.154.0

RUN wget -q \
  # This file disappeared from the canonical source:
  # "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION?}/linux/vulkansdk-linux-${VULKAN_SDK_VERSION?}.tar.gz"
  "https://storage.googleapis.com/iree-shared-files/vulkansdk-linux-${VULKAN_SDK_VERSION?}.tar.gz" \
  && mkdir -p /opt/vulkan-sdk \
  && tar -xzf "vulkansdk-linux-${VULKAN_SDK_VERSION?}.tar.gz" -C /opt/vulkan-sdk \
  && rm -rf /install-vulkan
WORKDIR /

ENV VULKAN_SDK="/opt/vulkan-sdk/${VULKAN_SDK_VERSION}/x86_64"

ENV PATH="${VULKAN_SDK}/bin:$PATH"

# Symlink the Vulkan loader to a system library directory. This is needed to
# allow Vulkan applications to find the Vulkan loader. It also avoids using
# LD_LIBRARY_PATH, which is not supported well by Docker.
RUN ln -s "${VULKAN_SDK}/lib/libvulkan.so" /usr/lib/x86_64-linux-gnu/ \
  && ln -s "${VULKAN_SDK}/lib/libvulkan.so.1" /usr/lib/x86_64-linux-gnu/

############## \
