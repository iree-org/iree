#!/bin/bash

# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Checks that Vulkan is working correctly and logs some useful information

set -xeuo pipefail

vulkaninfo 2> /tmp/vulkaninfo.stderr 1> /tmp/vulkaninfo.stdout

VULKAN_INSTANCE="$(grep "Vulkan Instance" /tmp/vulkaninfo.stdout)"

if [[ -z "${VULKAN_INSTANCE?}" ]]; then
  echo "Vulkan not found!"
  cat /tmp/vulkaninfo.stdout
  cat /tmp/vulkaninfo.stderr
  exit 1
fi

cat /tmp/vulkaninfo.stdout
cat /tmp/vulkaninfo.stderr
