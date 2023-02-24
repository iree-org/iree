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
    "py_module.h"
    "py_module.cc"
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
    iree::base::internal::flags
    iree::base::tracing
    iree::hal
    iree::hal::drivers
    iree::hal::utils::allocators
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
    "iree/runtime/benchmark.py"
    "iree/runtime/flags.py"
    "iree/runtime/function.py"
    "iree/runtime/system_api.py"
    "iree/runtime/system_setup.py"
    "iree/runtime/tracing.py"
    "iree/runtime/scripts/iree_benchmark_trace/__main__.py"
    "iree/runtime/scripts/iree_benchmark_module/__main__.py"
    "iree/runtime/scripts/iree_run_trace/__main__.py"
    "iree/runtime/scripts/iree_run_module/__main__.py"
    ${_PYTHON_EXTRA_SRCS}
  PYEXT_DEPS
    ::PyExtRt
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree-benchmark-module
  TO_EXE_NAME iree/runtime/iree-benchmark-module
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree-benchmark-trace
  TO_EXE_NAME iree/runtime/iree-benchmark-trace
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree-run-trace
  TO_EXE_NAME iree/runtime/iree-run-trace
)

iree_symlink_tool(
  TARGET runtime
  FROM_TOOL_TARGET iree-run-module
  TO_EXE_NAME iree/runtime/iree-run-module
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
    py_module_test
  SRCS
    "tests/py_module_test.py"
)

iree_py_test(
  NAME
    system_api_test
  SRCS
    "tests/system_api_test.py"
)

iree_py_test(
  NAME
    system_setup_test
  SRCS
    "tests/system_setup_test.py"
)

iree_py_test(
  NAME
    vm_test
  SRCS
    "tests/vm_test.py"
)

iree_py_test(
  NAME
    vm_types_test
  SRCS
    "tests/vm_types_test.py"
)

# TODO: Enable this once the CI bots are updated to install the python3-venv
# apt package. https://github.com/openxla/iree/issues/9080
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

set(_INSTALL_DIR "python_packages/iree_runtime")
set(_INSTALL_COMPONENT "IreePythonPackage-runtime")

# Install iree/runtime/*.py files verbatim into the tree.
# We do this at the package level so as to avoid any loose files
# from outside (i.e. tests, etc).
install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/iree/runtime/"
  COMPONENT "${_INSTALL_COMPONENT}"
  DESTINATION "${_INSTALL_DIR}/iree/runtime/"
  FILES_MATCHING PATTERN "*.py"
)

# _runtime.so -> python_packages/iree_runtime/iree/_runtime.so
install(
  TARGETS iree_runtime_bindings_python_PyExtRt
  DESTINATION "${_INSTALL_DIR}/iree"
  COMPONENT "${_INSTALL_COMPONENT}"
)

# Install tools into python_packages/iree_runtime/iree/runtime
#
# Our runtime/... directory is included by the root CMakeLists before the
# tools/ directory which defines these targets, so we defer the install() to
# the end of the root file. While deferred calls are generally fragile, this
# install is purely a leaf feature (with no other calls depending on its
# sequencing), so this use is okay.
cmake_language(EVAL CODE "
cmake_language(DEFER DIRECTORY \"${IREE_SOURCE_DIR}\"
  CALL install
  TARGETS
    iree-benchmark-module
    iree-benchmark-trace
    iree-run-module
    iree-run-trace
    ${_EXTRA_INSTALL_TOOL_TARGETS}
  DESTINATION \"${_INSTALL_DIR}/iree/runtime\"
  COMPONENT \"${_INSTALL_COMPONENT}\"
)
")
