# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Hand-written: bazel_to_cmake cannot express the test environment below.

if(IREE_COMPILER_DYNAMIC_PLUGINS)
  iree_lit_test_suite(
    NAME
      lit
    SRCS
      "annotate.mlir"
    TOOLS
      FileCheck
      iree-compile
  )

  # Nothing to attach to when tests are off: the suite above creates no tests.
  if(IREE_BUILD_TESTS)
    iree_package_ns(_PACKAGE_NS)
    string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
    set_property(TEST "${_PACKAGE_PATH}/annotate.mlir.test"
      APPEND PROPERTY ENVIRONMENT
      "IREE_OOTEX_PLUGIN=$<TARGET_FILE:iree_compiler_plugin_ootex>"
    )
  endif()
endif()
