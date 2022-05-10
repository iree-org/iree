# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(NUMPY_DEPS "")
set(PYBIND_COPTS "-fexceptions")
set(PYBIND_EXTENSION_COPTS "-fvisibility=hidden")

set(_PYTHON_EXTRA_SRCS)
set(_EXTRA_INSTALL_TOOL_TARGETS)
set(_TRACY_ENABLED OFF)

if(TARGET IREETracyCaptureServer)
  message(STATUS "Bundline Tracy CLI tools with Python API")
  set(_TRACY_ENABLED ON)
  list(APPEND _PYTHON_EXTRA_SRCS "iree/runtime/scripts/iree-tracy-capture")
  list(APPEND _EXTRA_INSTALL_TOOL_TARGETS "IREETracyCaptureServer")
endif()

################################################################################
# Package
################################################################################

iree_pyext_module(
  NAME
    PyExtRt
  MODULE_NAME iree/_runtime
  SRCS
    "binding.h"
    "initialize_module.cc"
    "invoke.h"
    "invoke.cc"
    "hal.h"
    "hal.cc"
    "status_utils.cc"
    "status_utils.h"
    "vm.h"
    "vm.cc"
  UNIX_LINKER_SCRIPT
    "unix_version.lds"
  DEFINES
    # Pybind code seems to be incompatible with C++ allocation tracing
    # hooks so disable it.
    IREE_TRACING_HOOK_CPP_NEW_DELETE=0
  DEPS
    iree::base
    iree::base::cc
    iree::base::internal::flags
    iree::base::tracing
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
    "iree/runtime/__init__.py"
    "iree/runtime/_binding.py"
    "iree/runtime/array_interop.py"
    "iree/runtime/flags.py"
    "iree/runtime/function.py"
    "iree/runtime/system_api.py"
    "iree/runtime/tracing.py"
    "iree/runtime/scripts/iree_benchmark_trace/__main__.py"
    "iree/runtime/scripts/iree_run_trace/__main__.py"
    "iree/runtime/scripts/iree_run_module/__main__.py"
    ${_PYTHON_EXTRA_SRCS}
  PYEXT_DEPS
    ::PyExtRt
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree_tools_iree-benchmark-module
  TO_EXE_NAME iree-benchmark-module
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

if(_TRACY_ENABLED)
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
    "tests/array_interop_test.py"
)

iree_py_test(
  NAME
    flags_test
  SRCS
    "tests/flags_test.py"
)

iree_py_test(
  NAME
    function_test
  SRCS
    "tests/function_test.py"
)

iree_py_test(
  NAME
    hal_test
  SRCS
    "tests/hal_test.py"
)

iree_py_test(
  NAME
    system_api_test
  SRCS
    "tests/system_api_test.py"
)

iree_py_test(
  NAME
    vm_test
  SRCS
    "tests/vm_test.py"
)

# TODO: Enable this once the CI bots are updated to install the python3-venv
# apt package. https://github.com/google/iree/issues/9080
# iree_py_test(
#   NAME
#     package_test
#   SRCS
#     "tests/package_test.py"
#   ARGS
#     "${IREE_BINARY_DIR}/runtime"
# )

################################################################################
# Install
################################################################################

iree_py_install_package(
  COMPONENT IreePythonPackage-runtime
  PACKAGE_NAME iree_runtime
  MODULE_PATH iree/runtime
  DEPS
    # TODO: Update CMake target/path mangling rules to make this syntactically
    # rooted on "iree" in some way.
    runtime_bindings_python_PyExtRt
    iree_tools_iree-benchmark-module
    iree_tools_iree-benchmark-trace
    iree_tools_iree-run-module
    iree_tools_iree-run-trace
    ${_EXTRA_INSTALL_TOOL_TARGETS}
  ADDL_PACKAGE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/README.md
)

install(
  # TODO: Update CMake target/path mangling rules to make this syntactically
  # rooted on "iree" in some way.
  TARGETS runtime_bindings_python_PyExtRt
  COMPONENT ${PY_INSTALL_COMPONENT}
  DESTINATION "${PY_INSTALL_MODULE_DIR}"
)

install(
  TARGETS
    iree_tools_iree-benchmark-module
    iree_tools_iree-benchmark-trace
    iree_tools_iree-run-module
    iree_tools_iree-run-trace
    ${_EXTRA_INSTALL_TOOL_TARGETS}
  DESTINATION "${PY_INSTALL_MODULE_DIR}"
  COMPONENT "${PY_INSTALL_COMPONENT}"
)
