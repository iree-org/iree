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

FROM gcr.io/iree-oss/util@sha256:40846b4aea5886af3250399d6adfdb3e1195a8b0177706bb0375e812d62dc49c AS base

ARG VULKAN_SDK_VERSION=1.2.154.0

RUN wget -q \
  "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION?}/linux/vulkansdk-linux-${VULKAN_SDK_VERSION?}.tar.gz" \
  -O "/tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz"

RUN mkdir -p /opt/vulkan-sdk

RUN tar -xf /tmp/vulkansdk-linux-x86_64-$VULKAN_SDK_VERSION.tar.gz -C /opt/vulkan-sdk
