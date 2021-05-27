# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_py_library(
  NAME
    compiler
  SRCS
    "__init__.py"
    "core.py"
    "tf.py"
    "tflite.py"
    "tools.py"
    "xla.py"
)

iree_py_install_package(
  COMPONENT IreePythonPackage-compiler
  PACKAGE_NAME iree_compiler
  MODULE_PATH iree/compiler
)
