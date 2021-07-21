# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_srcs
  "__init__.py"
  "scripts/iree_import_tflite/__main__.py"
)

iree_py_library(
  NAME
    tflite
  SRCS ${_srcs}
  DEPS
    iree_tf_compiler_iree-import-tflite
)

iree_symlink_tool(
  TARGET tflite
  FROM_TOOL_TARGET iree_tf_compiler_iree-import-tflite
  TO_EXE_NAME iree-import-tflite
)

iree_py_install_package(
  COMPONENT IreePythonPackage-tools-tflite
  PACKAGE_NAME iree_tools_tflite
  MODULE_PATH iree/tools/tflite
  FILES_MATCHING ${_srcs}
  DEPS
    iree_tf_compiler_iree-import-tflite
)

# Since imported, need to resolve the TARGET_FILE ourselves instead of
# install TARGETS form.
install(
  PROGRAMS "$<TARGET_FILE:iree_tf_compiler_iree-import-tflite>"
  DESTINATION "${PY_INSTALL_MODULE_DIR}"
  COMPONENT "${PY_INSTALL_COMPONENT}"
)
