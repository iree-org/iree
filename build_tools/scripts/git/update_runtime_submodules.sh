#!/bin/bash

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Initialize a subset of submodules for runtime only.

set -xeuo pipefail

RUNTIME_SUBMODULES=(
    "third_party/benchmark"
    "third_party/cpuinfo"
    "third_party/flatcc"
    "third_party/googletest"
    "third_party/liburing"
    "third_party/libyaml"
    "third_party/musl"
    "third_party/spirv_cross"
    "third_party/spirv_headers"
    "third_party/tracy"
    "third_party/vulkan_headers"
    "third_party/vulkan_memory_allocator"
    "third_party/webgpu-headers"
)

git submodule update --init ${RUNTIME_SUBMODULES[@]}
