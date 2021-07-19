# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT ${IREE_HAL_DRIVER_WEBGPU_NATIVE})
  return()
endif()

iree_cc_library(
  NAME
    wgpu
  HDRS
    "wgpu_driver.h"
  SRCS
    "wgpu_driver.c"
    "wgpu_util.c"
  DEPS
    iree::base
    iree::base::core_headers
    iree::base::internal
    iree::base::tracing
    iree::hal
    iree::hal::webgpu::platform
    wgpu-native-static
  PUBLIC
)
