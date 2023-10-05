# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "")

add_library(iree_pjrt_compiler_defs INTERFACE)
target_include_directories(iree_pjrt_compiler_defs
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Configures all iree_cc_* targets to take this implicit dep,
# which provides common includes and copts for the tree.
set(IREE_IMPLICIT_DEFS_CC_DEPS iree_pjrt_compiler_defs)

add_subdirectory(iree_pjrt/common)
add_subdirectory(iree_pjrt/partitioner_api)

if(IREE_HAL_DRIVER_LOCAL_TASK)
  add_subdirectory(iree_pjrt/cpu)
endif()
if(IREE_HAL_DRIVER_CUDA)
  add_subdirectory(iree_pjrt/cuda)
endif()
if("ROCM" IN_LIST IREE_EXTERNAL_HAL_DRIVERS)
  add_subdirectory(iree_pjrt/rocm)
endif()
if(IREE_HAL_DRIVER_VULKAN)
  add_subdirectory(iree_pjrt/vulkan)
endif()
