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

# An image for building IREE using CMake and testing IREE with SwiftShader
# Vulkan implementation.

# Build using:
# docker build --tag gcr.io/iree-oss/cmake-swiftshader \
#   build_tools/docker/cmake_swiftshader/

# Run interactively using the following, where IREE_WORKDIR is the path to your
# local dev environment:
# docker run -it --rm --entrypoint bash \
#   --volume "${IREE_WORKDIR?}:/usr/src/iree/" \
#   --gpus all \
#   gcr.io/iree-oss/cmake-swiftshader

# Set up the image and working directory by inheriting the base
# CMake configuration.
FROM gcr.io/iree-oss/cmake-vulkan

ARG SWIFTSHADER_COMMIT=6a8a74986c357b0c6fa0dfd2b4b9230af8d39d1a

# zlib is needed for compiling SwiftShader.
RUN apt-get update && apt-get install zlib1g-dev

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
