# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_LLVM_CPU OR
   NOT IREE_HAL_DRIVER_LOCAL_SYNC OR
   NOT IREE_BUILD_COMPILER)
  return()
endif()

## Example with VM bytecode module.
# Setup args for iree-compile.
set(_COMPILE_ARGS)
list(APPEND _COMPILE_ARGS "--iree-hal-target-backends=llvm-cpu")
list(APPEND _COMPILE_ARGS "--iree-llvmcpu-link-embedded=false")
list(APPEND _COMPILE_ARGS "--iree-llvmcpu-link-static")
list(APPEND _COMPILE_ARGS "--iree-llvmcpu-static-library-output-path=simple_mul.o")
list(APPEND _COMPILE_ARGS "--iree-vm-target-index-bits=32")
list(APPEND _COMPILE_ARGS "${CMAKE_CURRENT_SOURCE_DIR}/simple_mul.mlir")
list(APPEND _COMPILE_ARGS "-o")
list(APPEND _COMPILE_ARGS "simple_mul.vmfb")

# Custom command for iree-compile to generate static library and bytecode.
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/simple_mul.h
    ${CMAKE_CURRENT_BINARY_DIR}/simple_mul.o
    ${CMAKE_CURRENT_BINARY_DIR}/simple_mul.vmfb
  COMMAND iree-compile ${_COMPILE_ARGS}
  DEPENDS iree-compile "simple_mul.mlir"
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
# need to run iree-compile manually to produce the desired '.vmfb' and '.h/.o'
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
  SRCS
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
  "create_bytecode_module.c"
  "static_library_demo.c"
DEPS
  ::simple_mul_c
  iree::runtime
  iree::hal::drivers::local_sync::sync_driver
  iree::hal::local::loaders::static_library_loader
  simple_mul
)

iree_lit_test(
  NAME
    static_library_demo_test
  TEST_FILE
    "static_library_demo_test.txt"
  TOOLS
    ::static_library_demo
    FileCheck
  LABELS
    "hostonly"
)

if(NOT (IREE_OUTPUT_FORMAT_C OR IREE_HOST_BIN_DIR))
  return()
endif()

## Example with VM C module.
# Setup args for iree-compile.
set(_COMPILE_ARGS)
list(APPEND _COMPILE_ARGS "--output-format=vm-c")
list(APPEND _COMPILE_ARGS "--iree-hal-target-backends=llvm-cpu")
list(APPEND _COMPILE_ARGS "--iree-llvmcpu-link-embedded=false")
list(APPEND _COMPILE_ARGS "--iree-llvmcpu-link-static")
list(APPEND _COMPILE_ARGS "--iree-llvmcpu-static-library-output-path=simple_mul_c_module.o")
list(APPEND _COMPILE_ARGS "--iree-vm-target-index-bits=32")
list(APPEND _COMPILE_ARGS "${CMAKE_CURRENT_SOURCE_DIR}/simple_mul.mlir")
list(APPEND _COMPILE_ARGS "-o")
list(APPEND _COMPILE_ARGS "simple_mul_emitc.h")

# Custom command for iree-compile to generate static library and C module.
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/simple_mul_c_module.h
    ${CMAKE_CURRENT_BINARY_DIR}/simple_mul_c_module.o
    ${CMAKE_CURRENT_BINARY_DIR}/simple_mul_emitc.h
  COMMAND iree-compile ${_COMPILE_ARGS}
  DEPENDS iree-compile "simple_mul.mlir"
)

# TODO(marbre): Cleanup custom targets and libraries.
add_custom_target(
  simple_mul_gen
  DEPENDS
    "simple_mul_emitc.h"
)

add_library(simple_mul_c_module
  STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/simple_mul_c_module.o
)
add_dependencies(simple_mul_c_module simple_mul_gen)

SET_TARGET_PROPERTIES(
  simple_mul_c_module
  PROPERTIES
  LINKER_LANGUAGE C
)

# TODO(marbre): Cleanup SRCS and DEPS.
iree_cc_library(
  NAME
    simple_mul_emitc
  HDRS
    "simple_mul_emitc.h"
  DEFINES
    "EMITC_IMPLEMENTATION"
)

iree_cc_binary(
  NAME
    static_library_demo_c
  SRCS
    "create_c_module.c"
    "static_library_demo.c"
  DEPS
    ::simple_mul_emitc
    iree::runtime
    iree::hal::drivers::local_sync::sync_driver
    iree::hal::local::loaders::static_library_loader
    iree::vm::shims_emitc
    simple_mul_c_module
)

iree_lit_test(
  NAME
    static_library_demo_c_test
  TEST_FILE
    "static_library_demo_c_test.txt"
  TOOLS
    ::static_library_demo_c
    FileCheck
  LABELS
    "hostonly"
)
