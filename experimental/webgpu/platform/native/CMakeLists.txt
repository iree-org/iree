# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    native
  HDRS
    "native_driver.h"
  SRCS
    "native_driver.c"
  DEPS
    iree::base
    iree::base::core_headers
    iree::base::internal
    iree::base::tracing
    iree::hal
    iree::experimental::webgpu::platform
  PUBLIC
)
