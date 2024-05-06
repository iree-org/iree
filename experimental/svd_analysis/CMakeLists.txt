# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    svd_analysis
  SHARED
  SRCS
    svd_analysis.c
    svd_analysis_impl.cc
  DEPS
    iree::base::core_headers
    iree::builtins::ukernel::internal_headers
  INCLUDES
    "${EIGEN3_INCLUDE_DIR}"
  COPTS
    "-fno-fast-math"
    "-march=native"
    "-Wno-unused-variable"
)
