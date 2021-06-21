# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_srcs
  "__init__.py"
  "scripts/iree_translate/__main__.py"
)

iree_py_library(
  NAME
    core
  SRCS ${_srcs}
)

if(${IREE_BUILD_COMPILER})
  iree_symlink_tool(
    TARGET core
    FROM_TOOL_TARGET iree_tools_iree-translate
    TO_EXE_NAME iree-translate
  )
endif()

iree_py_install_package(
  AUGMENT_EXISTING_PACKAGE
  COMPONENT IreePythonPackage-compiler
  PACKAGE_NAME iree_compiler
  MODULE_PATH iree/tools/core
  FILES_MATCHING ${_srcs}
  DEPS
    iree_tools_iree-translate
)

install(
  TARGETS iree_tools_iree-translate
  COMPONENT ${PY_INSTALL_COMPONENT}
  DESTINATION "${PY_INSTALL_MODULE_DIR}"
)
