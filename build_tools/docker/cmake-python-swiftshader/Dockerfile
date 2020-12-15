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

FROM gcr.io/iree-oss/cmake-python-vulkan@sha256:6722f69c6300749f6bd4b141fc653244990381a6b0111f9c361061adcd65c07c AS final
COPY --from=gcr.io/iree-oss/swiftshader@sha256:ccae32c83c89a31e8fc5542e480c29f28bbf4a3b3b80198c06b687a92c6813f3 \
  /swiftshader /swiftshader

# Set VK_ICD_FILENAMES so Vulkan loader can find the SwiftShader ICD.
ENV VK_ICD_FILENAMES /swiftshader/vk_swiftshader_icd.json
