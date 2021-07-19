# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT EMSCRIPTEN)
  return()
endif()

iree_cc_library(
  NAME
    emscripten
  HDRS
    "emscripten_driver.h"
  SRCS
    "emscripten_driver.c"
    "emscripten_util.c"
  DEPS
    iree::base
    iree::base::core_headers
    iree::base::internal
    iree::base::tracing
    iree::hal
    iree::experimental::webgpu::platform
  PUBLIC
)
