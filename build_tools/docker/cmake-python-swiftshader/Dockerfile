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

FROM gcr.io/iree-oss/cmake-python-vulkan AS final
COPY --from=gcr.io/iree-oss/swiftshader /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json

COPY --from=gcr.io/iree-oss/vulkan /opt/vulkan-sdk/ /opt/vulkan-sdk/
ARG VULKAN_SDK_VERSION=1.2.154.0
ENV VULKAN_SDK="/opt/vulkan-sdk/${VULKAN_SDK_VERSION}/x86_64"
ENV PATH="${VULKAN_SDK}/bin:$PATH"
RUN ln -s "${VULKAN_SDK}/lib/libvulkan.so" /usr/lib/x86_64-linux-gnu/ \
  && ln -s "${VULKAN_SDK}/lib/libvulkan.so.1" /usr/lib/x86_64-linux-gnu/
