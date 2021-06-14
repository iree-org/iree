# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_py_library(
  NAME
    jax
  SRCS
    "__init__.py"
    "frontend.py"
)

# Only enable the tests if the XLA compiler is built.
if(${IREE_BUILD_XLA_COMPILER})
iree_py_test(
  NAME
    frontend_test
  SRCS
    "frontend_test.py"
)
endif()

iree_py_install_package(
  COMPONENT IreePythonPackage-jax
  PACKAGE_NAME iree_jax
  MODULE_PATH iree/jax
  ADDL_PACKAGE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/README.md
)
