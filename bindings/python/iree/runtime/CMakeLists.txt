# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_pyext_module(
  NAME
    PyExtRt
  MODULE_NAME binding
  SRCS
    "initialize_module.cc"
    "binding.h"
    "hal.h"
    "vm.h"
    "hal.cc"
    "status_utils.cc"
    "status_utils.h"
    "vm.cc"
  UNIX_LINKER_SCRIPT
    "unix_version.lds"
  DEPS
    iree::base
    iree::base::cc
    iree::hal
    iree::hal::drivers
    iree::modules::hal
    iree::vm
    iree::vm::bytecode_module
)

iree_py_library(
  NAME
    runtime
  SRCS
    "__init__.py"
    "function.py"
    "system_api.py"
    "tracing.py"
  PYEXT_DEPS
    ::PyExtRt
)

iree_py_test(
  NAME
    function_test
  SRCS
    "function_test.py"
)

iree_py_test(
  NAME
    hal_test
  SRCS
    "hal_test.py"
)

iree_py_test(
  NAME
    system_api_test
  SRCS
    "system_api_test.py"
)

iree_py_test(
  NAME
    vm_test
  SRCS
    "vm_test.py"
)

iree_py_install_package(
  COMPONENT IreePythonPackage-runtime
  PACKAGE_NAME iree_runtime
  MODULE_PATH iree/runtime
  DEPS
    bindings_python_iree_runtime_PyExtRt
  ADDL_PACKAGE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/README.md
)

install(
  TARGETS bindings_python_iree_runtime_PyExtRt
  COMPONENT ${PY_INSTALL_COMPONENT}
  DESTINATION "${PY_INSTALL_MODULE_DIR}"
)
