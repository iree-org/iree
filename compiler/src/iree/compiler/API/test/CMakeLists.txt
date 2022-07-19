# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_BUILD_TESTS)
  return()
endif()

iree_cc_binary(
  NAME
    api-test-binary
  SRCS
    "api-test-main.c"
  DEPS
    IREECompilerCAPILib
    MLIRCAPIIR
    iree::base
  EXCLUDE_FROM_ALL
)

add_dependencies(iree-test-deps "iree_compiler_API_test_api-test-binary")

iree_native_test(
  NAME
    "api-test"
  SRC
    ::api-test-binary
)
