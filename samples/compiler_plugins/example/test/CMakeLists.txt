# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Hand-written: bazel_to_cmake cannot express the test environment below.

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "static_link.mlir"
  TOOLS
    FileCheck
    iree-compile
)

if(IREE_COMPILER_DYNAMIC_PLUGINS)
  iree_lit_test_suite(
    NAME
      lit_dynamic
    SRCS
      "dynamic_link.mlir"
    TOOLS
      FileCheck
      iree-compile
  )

  # samples/lit.cfg.py forwards every IREE_* variable, and gates the
  # iree_dynamic_plugins feature on this one. The name is recomputed the way
  # iree_lit_test does so the two cannot drift.
  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set_property(TEST "${_PACKAGE_PATH}/dynamic_link.mlir.test"
    APPEND PROPERTY ENVIRONMENT
    "IREE_EXAMPLE_DYN_PLUGIN=$<TARGET_FILE:iree_compiler_plugin_example_dyn>"
  )
endif()
