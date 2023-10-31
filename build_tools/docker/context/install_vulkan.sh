#!/bin/bash
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

VULKAN_SDK_VERSION=$1

ARCH="$(uname -m)"
if [[ "${ARCH}" == "x86_64" ]]; then
  wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - \
    && wget -qO /etc/apt/sources.list.d/lunarg-vulkan-${VULKAN_SDK_VERSION}-focal.list https://packages.lunarg.com/vulkan/${VULKAN_SDK_VERSION}/lunarg-vulkan-${VULKAN_SDK_VERSION}-focal.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends vulkan-sdk
else
  echo "Installing Vulkan for ${ARCH} is not supported yet."
fi
