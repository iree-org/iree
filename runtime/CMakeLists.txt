# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_enable_optimization_options(
  PROFILE_NAME
    "${IREE_RUNTIME_OPTIMIZATION_PROFILE}"
  # TODO: These options should be separated between those required as
  # INTERFACE and those that can be private (i.e. to the runtime).
  SIZE_INTERFACE_COPTS
    "-DIREE_STATUS_MODE=0"
    "-DIREE_STATISTICS_ENABLE=0"
    "-DIREE_HAL_MODULE_STRING_UTIL_ENABLE=0"
    "-DIREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE=0"
    "-DIREE_VM_BACKTRACE_ENABLE=0"
    "-DIREE_VM_BYTECODE_VERIFICATION_ENABLE=0"
    "-DIREE_VM_EXT_F32_ENABLE=0"
    "-DIREE_VM_EXT_F64_ENABLE=0"
)

# Must include runtime plugins before processing the runtime sources so that
# the static link list can be set.
iree_include_cmake_plugin_dirs(
  LOG_LABEL
    runtime
  BINARY_DIR
    "${IREE_BINARY_DIR}/runtime/plugins"
  PLUGIN_CMAKE_FILE
    "iree_runtime_plugin.cmake"
)

add_subdirectory(src)

if(IREE_BUILD_PYTHON_BINDINGS)
  add_subdirectory(bindings/python)
endif()
