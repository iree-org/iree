#!/bin/bash

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

# For use within a IREE bazel-swiftshader docker image on a Kokoro VM.
# Log some information about the environment, initialize the submodules and then
# run the bazel integrations tests.

set -e
set -x

# Print Vulkan related information: SDK version and GPU ICD version
vulkaninfo 2> /tmp/vulkaninfo.stderr 1> /tmp/vulkaninfo.stdout
VULKAN_INSTANCE="$(grep "Vulkan Instance" /tmp/vulkaninfo.stdout)"
VK_PHYSICAL_DEVICE_PROPERTIES="$(grep -A7 "VkPhysicalDeviceProperties" /tmp/vulkaninfo.stdout)"

if [[ -z "${VULKAN_INSTANCE?}" ]] || [[ -z "${VK_PHYSICAL_DEVICE_PROPERTIES?}" ]]; then
  echo "Vulkan not found!"
  cat /tmp/vulkaninfo.stdout
  cat /tmp/vulkaninfo.stderr
  exit 1
fi

echo "${VULKAN_INSTANCE?}"
echo "${VK_PHYSICAL_DEVICE_PROPERTIES?}"
