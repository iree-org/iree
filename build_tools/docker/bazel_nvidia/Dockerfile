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

# An image for building IREE's tensorflow integrations using bazel and vulkan.

# Build using:
# docker build --tag gcr.io/iree-oss/bazel-nvidia \
#   build_tools/docker/bazel_nvidia/

# Run interactively using the following, where IREE_WORKDIR is the path to your
# local dev environment:
# docker run -it --rm --entrypoint bash --volume ${IREE_WORKDIR}:/usr/src/iree/ \
#   gcr.io/iree-oss/bazel-nvidia

# Set up the image and working directory.
# We start from bazel-nvidia so this image can be used to testing TensorFlow
# integrations.
FROM gcr.io/iree-oss/bazel-tensorflow

# Additionally, we need to install the Vulkan SDK and the NVIDIA Vulkan driver.

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
  && DEBIAN_FRONTEND=noninteractive apt-get install -y vulkan-sdk nvidia-driver-440
