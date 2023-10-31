# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    client
  HDRS
    "client.h"
  SRCS
    "client.cc"
  DEPS
    iree_pjrt::common
    iree::hal::drivers::cuda
)

iree_cc_library(
  SHARED
  NAME
    dylib
  DEFINES
    # Causes PJRT dynamic linking entry points to be made visible.
    PJRT_PLUGIN_BUILDING_LIBRARY
  SRCS
    "dylib_entry_point.cc"
  DEPS
    ::client
    iree_pjrt::common
    iree_pjrt::common::dylib_platform
)

# Output to the project wide python binary directory tree.
set(_NATIVE_PYTHON_DIR "${IREE_PJRT_PYTHON_BINARY_DIR}/iree/_pjrt_libs/cuda")
file(WRITE "${_NATIVE_PYTHON_DIR}/__init__.py" "")
set_target_properties(iree_pjrt_cuda_dylib
  PROPERTIES
    PREFIX ""  # Disable "lib" prefix.
    LIBRARY_OUTPUT_NAME pjrt_plugin_iree_cuda
    RUNTIME_OUTPUT_DIRECTORY "${_NATIVE_PYTHON_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${_NATIVE_PYTHON_DIR}"
)

# TODO: Find a better way to decide whether can link with undefined symbols.
if(NOT IREE_ENABLE_ASAN)
  if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_link_options(iree_pjrt_cpu_dylib PRIVATE "-Wl,-undefined,error")
  else()
    target_link_options(iree_pjrt_cpu_dylib PRIVATE "-Wl,--no-undefined")
  endif()
endif()
