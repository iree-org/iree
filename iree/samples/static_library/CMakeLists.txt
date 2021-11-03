# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT ${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT}
  OR NOT ${IREE_HAL_DRIVER_DYLIB}
  OR NOT ${IREE_BUILD_COMPILER})
  return()
endif()

# Set iree-translate binary.
set(_TRANSLATE_TOOL_EXECUTABLE $<TARGET_FILE:iree_tools_iree-translate>)

# Setup args for iree-translate.
set(_TRANSLATE_ARGS)
list(APPEND _TRANSLATE_ARGS "-iree-mlir-to-vm-bytecode-module")
list(APPEND _TRANSLATE_ARGS "-iree-hal-target-backends=dylib-llvm-aot")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-link-embedded=false")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-link-static")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-static-library-output-path=simple_mul.o")
list(APPEND _TRANSLATE_ARGS "${CMAKE_CURRENT_SOURCE_DIR}/simple_mul.mlir")
list(APPEND _TRANSLATE_ARGS "-o")
list(APPEND _TRANSLATE_ARGS "simple_mul.vmfb")

# Custom command for iree-translate to generate static library and bytecode.
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/simple_mul.h
    ${CMAKE_CURRENT_BINARY_DIR}/simple_mul.o
    ${CMAKE_CURRENT_BINARY_DIR}/simple_mul.vmfb
  COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${_TRANSLATE_ARGS}
  DEPENDS ${_TRANSLATE_TOOL_EXECUTABLE} "simple_mul.mlir"
)

# Tell cmake about the simple_mul library so it will link it.
add_library(simple_mul
  STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/simple_mul.o)

SET_TARGET_PROPERTIES(
  simple_mul
  PROPERTIES
  LINKER_LANGUAGE C)

# Note: If you're cross compiling the simple_mul for a different backend, you'll
# need to run iree-translate manually to produce the desired '.vmfb' and '.h/.o'
# files. Substitute the 'simple_mul' dependency in iree_cc_binary() below with
# your own static library and the `simple_mul.vmfb` in the iree_c_embed_data()
# rule. You can use paths to files, i.e. 'path/to/generated/output.vmfb' in
# place of CMake targets.

# Generate the embed data with the bytecode module
iree_c_embed_data(
  NAME
    simple_mul_c
  IDENTIFIER
    iree_samples_static_library_simple_mul
  GENERATED_SRCS
    simple_mul.vmfb
  C_FILE_OUTPUT
    simple_mul_c.c
  H_FILE_OUTPUT
    simple_mul_c.h
  FLATTEN
  PUBLIC
)

iree_cc_binary(
NAME
  static_library_demo
SRCS
  "static_library_demo.c"
DEPS
  ::simple_mul_c
  iree::runtime
  iree::hal::local::loaders::static_library_loader
  iree::hal::local::task_driver
  iree::task::api
  simple_mul
)

iree_lit_test(
  NAME
    static_library_demo_test
  TEST_FILE
    "static_library_demo_test.txt"
  DATA
    ::static_library_demo
    iree::tools::IreeFileCheck
  LABELS
    "hostonly"
)
