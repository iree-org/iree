# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

if(NOT IREE_HAL_DRIVER_CUDA)
  return()
endif()

iree_cc_library(
  NAME
    registration
  HDRS
    "driver_module.h"
  SRCS
    "driver_module.c"
  DEPS
    iree::base
    iree::base::cc
    iree::base::core_headers
    iree::base::internal::flags
    iree::base::tracing
    iree::hal
    iree::hal::cuda
  DEFINES
    "IREE_HAL_HAVE_CUDA_DRIVER_MODULE=1"
  PUBLIC
)
