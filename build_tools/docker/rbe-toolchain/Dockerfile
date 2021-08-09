# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image for building IREE in RBE's remmote execution environments.
# The parent image requires gcloud authorization to download (run
# `gcloud auth configure-docker`)

# Note that this doesn't reuse anything from our other docker images because it
# is based on Ubuntu 16.04, which is the only image that RBE provides toolchains
# for. There are also Ubuntu 18.04 images, but we'd have to construct all our
# own toolchains.

######################## Install Swiftshader ###################################
# Ubuntu 16.04
FROM ubuntu@sha256:3355b6e4ba1b12071ba5fe9742042a2f10b257c908fbdfac81912a16eb463879 AS install-swiftshader
WORKDIR /install-swiftshader

RUN apt-get update && apt-get install -y wget

ARG CMAKE_VERSION=3.13.5
# Install CMake v3.13, which is ahead of apt-get's version (3.10.2).
RUN wget "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION?}/cmake-${CMAKE_VERSION?}-Linux-x86_64.sh" \
  && chmod +x "./cmake-${CMAKE_VERSION?}-Linux-x86_64.sh" \
  && "./cmake-${CMAKE_VERSION?}-Linux-x86_64.sh" --skip-license --prefix=/usr/ \
  && rm "./cmake-${CMAKE_VERSION?}-Linux-x86_64.sh"

RUN apt-get update && apt-get install -y \
  clang \
  git \
  ninja-build \
  python3 \
  zlib1g-dev

# This commit fixed support for the max version of libstdc++6 available through
# Ubuntu 16.04 apt.
ARG SWIFTSHADER_COMMIT=84f5eeb6dd9b225f465f93737fa76aad7de355cf

RUN git clone https://github.com/google/swiftshader
RUN cd swiftshader && git checkout "${SWIFTSHADER_COMMIT?}" && cd ..
# Only build SwiftShader Vulkan ICD.
RUN cmake -S swiftshader/ -B build-swiftshader/ \
  -GNinja \
  -DSWIFTSHADER_BUILD_VULKAN=ON \
  -DSWIFTSHADER_BUILD_EGL=OFF \
  -DSWIFTSHADER_BUILD_GLESv2=OFF \
  -DSWIFTSHADER_BUILD_GLES_CM=OFF \
  -DSWIFTSHADER_BUILD_PVR=OFF \
  -DSWIFTSHADER_BUILD_TESTS=OFF
RUN cmake --build build-swiftshader/ \
  --config Release \
  --target vk_swiftshader

# Copy the ICD JSON and .so to a known place.
RUN cp -rf build-swiftshader/Linux /swiftshader
# Keep track of the commit we are using.
RUN echo "${SWIFTSHADER_COMMIT?}" > /swiftshader/git-commit

######################## Final Image ###########################################
FROM gcr.io/cloud-marketplace/google/rbe-ubuntu16-04@sha256:1a8ed713f40267bb51fe17de012fa631a20c52df818ccb317aaed2ee068dfc61 AS final

######################## Vulkan SDK ############################################

ARG VULKAN_SDK_VERSION=1.2.154.0

COPY --from=gcr.io/iree-oss/vulkan@sha256:5812ee64806a7f3df0739ccf0930c27cabce346901488eceb1ee66c9c0a5ae96 /opt/vulkan-sdk/ /opt/vulkan-sdk/

ENV VULKAN_SDK="/opt/vulkan-sdk/${VULKAN_SDK_VERSION}/x86_64"

ENV PATH="${VULKAN_SDK}/bin:$PATH"

# Symlink the Vulkan loader to a system library directory. This is needed to
# allow Vulkan applications to find the Vulkan loader. It also avoids using
# LD_LIBRARY_PATH, which is not supported well by Docker.
RUN ln -s "${VULKAN_SDK}/lib/libvulkan.so" /usr/lib/x86_64-linux-gnu/ \
  && ln -s "${VULKAN_SDK}/lib/libvulkan.so.1" /usr/lib/x86_64-linux-gnu/

######################## Swiftshader ###########################################
COPY --from=install-swiftshader /swiftshader /swiftshader
# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json


######################## LibXML2 for LLD #######################################
RUN apt-get update && apt-get install -y libxml2-dev
