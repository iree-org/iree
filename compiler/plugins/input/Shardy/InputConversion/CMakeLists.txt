# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    InputConversion
  HDRS
    "Passes.h"
  SRCS
    "Passes.cpp"
    "StripShardyDialect.cpp"
  DEPS
    MLIRIR
    MLIRPass
    MLIRTransformUtils
    SdyDialect
  PUBLIC
)
