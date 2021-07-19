# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(${IREE_HAL_DRIVER_WEBGPU_NATIVE})

iree_cc_library(
  NAME
    native
  HDRS
    "driver_module_native.h"
  SRCS
    "driver_module_native.c"
  DEPS
    iree::base
    iree::base::internal::flags
    iree::hal
    iree::hal::webgpu
    iree::hal::webgpu::platform::wgpu
  DEFINES
    "IREE_HAL_HAVE_WEBGPU_NATIVE_DRIVER_MODULE=1"
  PUBLIC
)

endif()
