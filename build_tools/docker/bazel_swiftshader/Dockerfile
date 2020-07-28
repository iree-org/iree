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

# An image for building IREE with tensorflow integrations using bazel and
# running Vulkan tests on SwiftShader.

# Build using:
# docker build --tag gcr.io/iree-oss/bazel-swiftshader \
#   build_tools/docker/bazel_swiftshader/

# Run interactively using the following, where IREE_WORKDIR is the path to your
# local dev environment:
# docker run -it --rm --entrypoint bash --volume ${IREE_WORKDIR}:/usr/src/iree/ \
#   gcr.io/iree-oss/bazel-swiftshader

# Set up the image and working directory.
FROM gcr.io/iree-oss/bazel-tensorflow

# TODO(#2651): The following steps are copied from cmake, cmake-vulkan, and
# cmake-swiftshader. We might want to consider using docker multi-stage
# builds to factor them out.

# Additionally, we need to install the Vulkan SDK.

ARG VULKAN_SDK_VERSION=1.2.141

# Disable apt-key parse waring. If someone knows how to do whatever the "proper"
# thing is then feel free. The warning complains about parsing apt-key output,
# which we're not even doing.
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

RUN wget -qO - http://packages.lunarg.com/lunarg-signing-key-pub.asc \
    | apt-key add - \
  && wget -qO \
    "/etc/apt/sources.list.d/lunarg-vulkan-${VULKAN_SDK_VERSION?}-bionic.list" \
    "http://packages.lunarg.com/vulkan/${VULKAN_SDK_VERSION?}/lunarg-vulkan-${VULKAN_SDK_VERSION?}-bionic.list" \
  && apt-get update \
  && apt-get install -y vulkan-sdk

ARG SWIFTSHADER_COMMIT=6a8a74986c357b0c6fa0dfd2b4b9230af8d39d1a

# Then compile and install SwiftShader.

# cmake, ninja, and zlib is needed for compiling SwiftShader.
RUN apt-get update && apt-get install -y cmake ninja-build zlib1g-dev

# Update cmake to v3.13+, which is ahead of apt-get's version (3.10.2).
ENV CMAKE_VERSION 3.13.5
RUN apt-get update \
  && mkdir ./cmake_install \
  && cd cmake_install \
  && wget "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION?}/cmake-${CMAKE_VERSION?}.tar.gz" \
  && tar -xzvf "cmake-${CMAKE_VERSION?}.tar.gz" \
  && cd "cmake-${CMAKE_VERSION?}/" \
  && cmake . \
  && make \
  && make install

RUN git clone https://github.com/google/swiftshader \
  && cd swiftshader && git checkout "${SWIFTSHADER_COMMIT?}" && cd .. \
  # Only build SwiftShader Vulkan ICD.
  && cmake -S swiftshader/ -B build-swiftshader/ \
           -GNinja \
           -DSWIFTSHADER_BUILD_VULKAN=ON \
           -DSWIFTSHADER_BUILD_EGL=OFF \
           -DSWIFTSHADER_BUILD_GLESv2=OFF \
           -DSWIFTSHADER_BUILD_GLES_CM=OFF \
           -DSWIFTSHADER_BUILD_PVR=OFF \
           -DSWIFTSHADER_BUILD_TESTS=OFF \
  && cmake --build build-swiftshader/ \
           --config Release \
           --target vk_swiftshader \
  # Copy the ICD JSON and .so to a known place.
  && cp -rf build-swiftshader/Linux /swiftshader \
  # Keep track of the commit we are using.
  && echo "${SWIFTSHADER_COMMIT?}" > /swiftshader/git-commit \
  # Clean up everything.
  && rm -rf swiftshader build-swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
