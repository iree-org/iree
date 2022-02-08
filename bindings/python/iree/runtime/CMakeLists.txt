# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_python_extra_srcs)
set(_extra_install_tool_targets)
set(_tracy_enabled OFF)

if(TARGET IREETracyCaptureServer)
  message(STATUS "Bundline Tracy CLI tools with Python API")
  set(_tracy_enabled ON)
  list(APPEND _python_extra_srcs "scripts/iree-tracy-capture")
  list(APPEND _extra_install_tool_targets "IREETracyCaptureServer")
endif()

################################################################################
# Package
################################################################################

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
    iree::base::internal::flags
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
    "array_interop.py"
    "flags.py"
    "function.py"
    "system_api.py"
    "tracing.py"
    "scripts/iree_benchmark_trace/__main__.py"
    "scripts/iree_run_trace/__main__.py"
    "scripts/iree_run_module/__main__.py"
    ${_python_extra_srcs}
  PYEXT_DEPS
    ::PyExtRt
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree_tools_iree-benchmark-trace
  TO_EXE_NAME iree-benchmark-trace
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree_tools_iree-run-trace
  TO_EXE_NAME iree-run-trace
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree_tools_iree-run-module
  TO_EXE_NAME iree-run-module
)

if(_tracy_enabled)
  iree_symlink_tool(
    TARGET runtime
    FROM_TOOL_TARGET IREETracyCaptureServer
    TO_EXE_NAME iree-tracy-capture
  )
endif()

################################################################################
# Tests
################################################################################

iree_py_test(
  NAME
    array_interop_test
  SRCS
    "array_interop_test.py"
)

iree_py_test(
  NAME
    flags_test
  SRCS
    "flags_test.py"
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

################################################################################
# Install
################################################################################

iree_py_install_package(
  COMPONENT IreePythonPackage-runtime
  PACKAGE_NAME iree_runtime
  MODULE_PATH iree/runtime
  DEPS
    bindings_python_iree_runtime_PyExtRt
    iree_tools_iree-benchmark-trace
    iree_tools_iree-run-module
    iree_tools_iree-run-trace
    ${_extra_install_tool_targets}
  ADDL_PACKAGE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/README.md
)

install(
  TARGETS bindings_python_iree_runtime_PyExtRt
  COMPONENT ${PY_INSTALL_COMPONENT}
  DESTINATION "${PY_INSTALL_MODULE_DIR}"
)

install(
  TARGETS
    iree_tools_iree-benchmark-trace
    iree_tools_iree-run-module
    iree_tools_iree-run-trace
    ${_extra_install_tool_targets}
  DESTINATION "${PY_INSTALL_MODULE_DIR}"
  COMPONENT "${PY_INSTALL_COMPONENT}"
)
