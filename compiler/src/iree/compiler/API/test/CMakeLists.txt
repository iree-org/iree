# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_binary(
  NAME
    compile-mhlo-test-binary
  SRCS
    "compile-mhlo-test-main.c"
  DEPS
    IREECompilerCAPILib
    MLIRCAPIIR
    iree::base
  TESTONLY
  EXCLUDE_FROM_ALL
)

add_dependencies(iree-test-deps "iree_compiler_API_test_compile-mhlo-test-binary")

iree_native_test(
  NAME
    "compile-mhlo-test"
  SRC
    ::compile-mhlo-test-binary
)
