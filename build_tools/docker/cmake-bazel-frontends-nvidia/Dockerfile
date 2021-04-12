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

# To use the host GPUs, `docker run` must be called with the `--gpus all` flag.

# Set up the image and working directory by inheriting the vulkan CMake
# configuration.
# Note that we don't start from NVIDIA's docker base:
# - nvidia/cuda (https://hub.docker.com/r/nvidia/cuda):
#     it's.. for CUDA.
# - nvidia/vulkan (https://hub.docker.com/r/nvidia/vulkan):
#      does not support Ubuntu 18.04.
# This allows to share configuration with base CMake, but it also means we need
# to MATCH the driver version between the host machine and the docker image.
FROM gcr.io/iree-oss/cmake-bazel-frontends-vulkan@sha256:02c21884ec62a548992fc62031c41375d5cb7ebc33cb4ecbb8ac0ec2c83a2aa1 AS final

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y \
       libnvidia-gl-460=460.39-0ubuntu0.18.04.1 \
       libnvidia-compute-460=460.39-0ubuntu0.18.04.1
