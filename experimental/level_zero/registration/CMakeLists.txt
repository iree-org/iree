# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
    iree::base::tracing
    iree::experimental::level_zero
    iree::hal
  DEFINES
    "IREE_HAVE_HAL_EXPERIMENTAL_LEVEL_ZERO_DRIVER_MODULE=1"
  PUBLIC
)
