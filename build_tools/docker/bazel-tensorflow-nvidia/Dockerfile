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

# An image for building IREE with TensorFlow integrations using Bazel and
# running Vulkan tests on an Nvidia GPU.

FROM gcr.io/iree-oss/bazel-tensorflow-vulkan@sha256:396a23a0bf14c07c33a9702ac11a4d551101bc0a9ab7896d259dccdc552fd723 AS final

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y vulkan-sdk nvidia-driver-440

# TODO: There's something screwy where rerunning apt-get breaks everything
# TF requires python and numpy at configure time...
RUN apt-get update \
    && apt-get install -y \
      python3 \
      python3-pip \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install numpy
