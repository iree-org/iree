# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FROM gcr.io/iree-oss/util@sha256:c9691ba3cd4287cee61be42103888d7c6f02c44c7c7b7d30a3dca7fc73ee92cb AS install-vulkan
WORKDIR /install-vulkan

ARG VULKAN_SDK_VERSION=1.2.154.0

RUN wget -q \
  # This file disappeared from the canonical source:
  # "https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION?}/linux/vulkansdk-linux-${VULKAN_SDK_VERSION?}.tar.gz"
  "https://storage.googleapis.com/iree-shared-files/vulkansdk-linux-${VULKAN_SDK_VERSION?}.tar.gz"

RUN mkdir -p /opt/vulkan-sdk

RUN tar -xzf "vulkansdk-linux-${VULKAN_SDK_VERSION?}.tar.gz" -C /opt/vulkan-sdk
