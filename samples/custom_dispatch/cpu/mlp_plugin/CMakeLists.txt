# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_LLVM_CPU OR
   NOT IREE_HAL_DRIVER_LOCAL_SYNC OR
   NOT IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
  return()
endif()

## Current support is only for x86.
if(NOT IREE_ARCH STREQUAL "x86_64")
  message(STATUS "IREE mlp_pluging sample ignored -- only builds for x86_64 (today)")
  return()
endif()

# system-library plugin mechanism using the system dynamic library loader.
if(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY)
  
add_library(iree_samples_custom_dispatch_cpu_mlp_plugin SHARED
  mlp_plugin.c
)
target_include_directories(iree_samples_custom_dispatch_cpu_mlp_plugin
  PRIVATE
    ${IREE_SOURCE_DIR}/runtime/src/
)

# NOTE: this is only required because we want this sample to run on all
# platforms without needing to change the library name (libfoo.so/foo.dll).
set_target_properties(iree_samples_custom_dispatch_cpu_mlp_plugin
  PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    PREFIX ""
    OUTPUT_NAME "mlp_plugin"
)

add_dependencies(iree-sample-deps
  iree_samples_custom_dispatch_cpu_mlp_plugin)

iree_lit_test_suite(
  NAME
    mlp_example
  SRCS
    "mlp.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
    iree_samples_custom_dispatch_cpu_mlp_plugin
  DATA
    "mlp_spec.mlir"
  LABELS
    "driver=local-sync"
    "hostonly"
)

iree_lit_test_suite(
  NAME
    mlp_tosa_example
  SRCS
    "mlp_tosa.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
    iree_samples_custom_dispatch_cpu_mlp_plugin
  DATA
    "mlp_tosa_spec.pdl.mlir"
  LABELS
    "driver=local-sync"
    "hostonly"
)

iree_lit_test_suite(
  NAME
    mlp_torch_example
  SRCS
    "mlp_torch.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
    iree_samples_custom_dispatch_cpu_mlp_plugin
  DATA
    "mlp_torch_spec.pdl.mlir"
  LABELS
    "driver=local-sync"
    "hostonly"
)

iree_lit_test_suite(
  NAME
    mlp_linalg_example
  SRCS
    "mlp_linalg.mlir"
    "mlp_linalg_two_matmul.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
    iree_samples_custom_dispatch_cpu_mlp_plugin
  DATA
    "mlp_linalg_spec.pdl.mlir"
  LABELS
    "driver=local-sync"
    "hostonly"
)

# Use `mlir-pdll` to build mlp_linglg_from_pdll.pdl.mlir from mpl_linalg_spec.pdll 
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/mlp_linalg_from_pdll.pdl.mlir"
  COMMAND mlir-pdll 
    "${CMAKE_CURRENT_SOURCE_DIR}/mlp_linalg_spec.pdll" 
    -I "${PROJECT_SOURCE_DIR}/third_party/llvm-project/mlir/include"
    -x=mlir 
   #-debug
    -o "${CMAKE_CURRENT_SOURCE_DIR}/mlp_linalg_from_pdll.pdl.mlir"
  DEPENDS
    mlir-pdll
    "${CMAKE_CURRENT_SOURCE_DIR}/mlp_linalg_spec.pdll"
  VERBATIM
)

add_custom_target(
  iree_samples_custom_dispatch_cpu_mlir_plugin_build_pdll ALL
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mlp_linalg_from_pdll.pdl.mlir"
)

add_dependencies(
  iree_samples_custom_dispatch_cpu_mlp_plugin 
  iree_samples_custom_dispatch_cpu_mlir_plugin_build_pdll
)

# Append the includes used for this file to the pdll_compilation_commands
file(APPEND ${CMAKE_BINARY_DIR}/pdll_compile_commands.yml
  "--- !FileInfo:\n"
  "  filepath: \"${CMAKE_CURRENT_SOURCE_DIR}/mlp_linalg_spec.pdll\"\n"
  "  includes: \"/home/ianwood/iree/third_party/llvm-project/mlir/include/\"\n"
)

iree_lit_test_suite(
  NAME
    mlp_pdll_linalg_example
  SRCS
    "mlp_linalg_pdll.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
    iree_samples_custom_dispatch_cpu_mlp_plugin
  DATA
    "mlp_linalg_from_pdll.pdl.mlir"
  LABELS
    "driver=local-sync"
    "hostonly"
)
endif(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY)
