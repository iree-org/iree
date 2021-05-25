# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(VMA_ROOT "${IREE_ROOT_DIR}/third_party/vulkan_memory_allocator/")

external_cc_library(
  PACKAGE
    vulkan_memory_allocator
  NAME
    vulkan_memory_allocator
  ROOT
    ${VMA_ROOT}
  HDRS
    "src/vk_mem_alloc.h"
  INCLUDES
    "${VMA_ROOT}/src/"
)
