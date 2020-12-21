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

FROM gcr.io/iree-oss/bazel-tensorflow-vulkan@sha256:b69ecaa70fde817bba4f01e47a77db8e7053d7d7b25ad8d4b1e47d370b055c3e AS final

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y libnvidia-gl-440=440.100-0ubuntu0.18.04.1
