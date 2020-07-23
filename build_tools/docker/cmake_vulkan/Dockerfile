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

# An image with Vulkan SDK for building IREE using CMake.

# Build using:
# docker build --tag gcr.io/iree-oss/cmake-vulkan \
#   build_tools/docker/cmake_vulkan/

# Run interactively using the following, where IREE_WORKDIR is the path to your
# local dev environment:
# docker run -it --rm --entrypoint bash \
#   --volume "${IREE_WORKDIR?}:/usr/src/iree/" \
#   --gpus all \
#   gcr.io/iree-oss/cmake-vulkan

# Set up the image and working directory by inheriting the base
# CMake configuration.
FROM gcr.io/iree-oss/cmake

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
