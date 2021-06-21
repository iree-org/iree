# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_srcs
  "__init__.py"
  "scripts/iree_import_tf/__main__.py"
)

iree_py_library(
  NAME
    tf
  SRCS ${_srcs}
  DEPS
  iree_tf_compiler_iree-import-tf
)

iree_symlink_tool(
  TARGET tf
  FROM_TOOL_TARGET iree_tf_compiler_iree-import-tf
  TO_EXE_NAME iree-import-tf
)

iree_py_install_package(
  COMPONENT IreePythonPackage-tools-tf
  PACKAGE_NAME iree_tools_tf
  MODULE_PATH iree/tools/tf
  FILES_MATCHING ${_srcs}
  DEPS
    iree_tf_compiler_iree-import-tf
)

# Since imported, need to resolve the TARGET_FILE ourselves instead of
# install TARGETS form.
install(
  PROGRAMS "$<TARGET_FILE:iree_tf_compiler_iree-import-tf>"
  DESTINATION "${PY_INSTALL_MODULE_DIR}"
  COMPONENT "${PY_INSTALL_COMPONENT}"
)
