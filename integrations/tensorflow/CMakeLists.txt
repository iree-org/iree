# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TensorFlow builds through bazel, and IREE maintains all of its TensorFlow
# dependent code under this directory tree. The CMake support is limited to
# compiler binaries and python bindings.
#
# If this directory is included, then building TensorFlow is assumed (the
# config option happens at the higher level).

add_subdirectory(iree_tf_compiler)

if(${IREE_BUILD_PYTHON_BINDINGS})
  add_subdirectory(bindings/python)
endif()

if(${IREE_BUILD_TESTS} AND ${IREE_BUILD_PYTHON_BINDINGS})
  add_subdirectory(test)
  if(${IREE_ENABLE_OLD_INTEGRATION_TESTS})
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/e2e/")
    include(iree_e2e_cartesian_product_test_suite)
    add_subdirectory(e2e)
  endif()
endif()
