# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Disabled on Windows until the shared library handling is cross-platform.
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  return()
endif()

# Only run if ROCM backend is enabled.
if(NOT IREE_TARGET_BACKEND_ROCM)
  return()
endif()

set(_NAME "iree_samples_compiler_plugins_rocm_opt_pass_plugin_ROCMPluginSample")

add_library(${_NAME} SHARED
  "ROCMPluginSample.cpp"
)
target_link_libraries(${_NAME}
  iree_compiler_API_SharedImpl
)

# NOTE: this is only required because we want this sample to run on all
# platforms without needing to change the library name (libfoo.so/foo.dll).
set_target_properties(${_NAME}
  PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    PREFIX ""
    OUTPUT_NAME "ROCMPluginSample"
)

target_compile_options(${_NAME} PRIVATE ${IREE_DEFAULT_COPTS})

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "rocm_plugin_sample.mlir"
  TOOLS
    FileCheck
    iree-opt
)
