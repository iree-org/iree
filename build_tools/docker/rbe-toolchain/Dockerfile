# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# An image for building IREE in RBE's remmote execution environments.
# The parent image requires gcloud authorization to download (run
# `gcloud auth configure-docker`)

# Note that this doesn't reuse anything from our other docker images because it
# is based on Ubuntu 16.04, which is the only image that RBE provides toolchains
# for. There are also Ubuntu 18.04 images, but we'd have to construct all our
# own toolchains.

######################## Install Swiftshader ###################################
FROM ubuntu:16.04 AS install-swiftshader
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

######################## Python 3 ##############################################
RUN apt-get update \
  && apt-get install -y \
    python3 \
    python3-pip \
  && python3 -m pip install --upgrade pip \
  && python3 -m pip install numpy

######################## Vulkan SDK ############################################

ARG VULKAN_SDK_VERSION=1.2.154.0

COPY --from=gcr.io/iree-oss/vulkan /opt/vulkan-sdk/ /opt/vulkan-sdk/

ENV VULKAN_SDK="/opt/vulkan-sdk/${VULKAN_SDK_VERSION}/x86_64"

######################## Swiftshader ###########################################
COPY --from=install-swiftshader /swiftshader /swiftshader
# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json

RUN apt-get update && apt-get install -y software-properties-common

# apt-add-repository requires a version of python with the softwareproperties
# module. To use this command, we:
#   1. remove the symlink to python3 from python3.6 and symlink it to python3.5
#   2. run apt-add-repository with python3 = python3.5
#   3. resymlink python3 to /opt/python3.6/bin/python3.6
# See https://github.com/google/iree/issues/1966 for more information.
RUN rm /usr/bin/python3 \
  && ln -s /usr/bin/python3.5 /usr/bin/python3 \
  && add-apt-repository ppa:deadsnakes/ppa \
  && rm /usr/bin/python3 \
  && ln -s /opt/python3.6/bin/python3.6 /usr/bin/python3

# Install python3.6-dev
RUN apt-get update \
  && apt-get install -y python3.6 python3.6-dev
