# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_HAL_DRIVER_MODULES)
if(IREE_HAL_DRIVER_CUDA)
  add_subdirectory(cuda)
  list(APPEND IREE_HAL_DRIVER_MODULES iree::hal::drivers::cuda::registration)
endif()
if(IREE_HAL_DRIVER_LOCAL_SYNC)
  add_subdirectory(local_sync)
  list(APPEND IREE_HAL_DRIVER_MODULES iree::hal::drivers::local_sync::registration)
endif()
if(IREE_HAL_DRIVER_LOCAL_TASK)
  add_subdirectory(local_task)
  list(APPEND IREE_HAL_DRIVER_MODULES iree::hal::drivers::local_task::registration)
endif()
if(IREE_HAL_DRIVER_VULKAN)
  add_subdirectory(vulkan)
  list(APPEND IREE_HAL_DRIVER_MODULES iree::hal::drivers::vulkan::registration)
endif()
if(IREE_HAL_DRIVER_EXPERIMENTAL_ROCM)
  list(APPEND IREE_HAL_DRIVER_MODULES experimental::rocm::registration)
endif()

iree_cc_library(
  NAME
    drivers
  HDRS
    "init.h"
  SRCS
    "init.c"
  DEPS
    iree::base
    iree::base::tracing
    ${IREE_HAL_DRIVER_MODULES}
  PUBLIC
)
