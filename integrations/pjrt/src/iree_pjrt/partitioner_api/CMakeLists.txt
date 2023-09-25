# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
    NAME
      partitioner_api
    HDRS
      "embedding_api.h"
      "loader.cpp"
    PUBLIC
)

iree_cc_library(
    NAME
      loader
    SRCS
      "loader/handle_symbols.inc"
      "loader/loader.cpp"
    DEPS
      ::partitioner_api
    PUBLIC
)
