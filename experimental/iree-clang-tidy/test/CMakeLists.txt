# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "test_file.c"
  TOOLS
    ${IREE_CLANG_TIDY_TARGET}
    FileCheck
    IREEClangTidyModule
    not
  LABELS
    "hostonly"
  TIMEOUT
    300
)
