# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_executable(
  compile-mhlo-test-binary
  compile-mhlo-test-main.c
)
target_link_libraries(compile-mhlo-test-binary
  PRIVATE
    IREECompilerCAPILib
    MLIRCAPIIR
)

add_test(
  NAME "iree/compiler/API/test/compile-mhlo-test"
  COMMAND compile-mhlo-test-binary
)
