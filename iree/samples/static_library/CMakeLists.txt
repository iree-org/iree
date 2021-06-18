# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# if(${IREE_BUILD_STATIC_LIBRARY_SAMPLES} AND NOT ${IREE_BUILD_SAMPLES})
#   return()
# endif()

if(NOT ${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT}
  OR NOT ${IREE_HAL_DRIVER_DYLIB}
  OR NOT ${IREE_BUILD_COMPILER}
  OR NOT ${IREE_BUILD_SAMPLES})
  return()
endif()

iree_bytecode_module(
  NAME
    simple_mul
  SRC
    "simple_mul.mlir"
  C_IDENTIFIER
    "iree_samples_static_library_simple_mul"
  FLAGS
    "-iree-input-type=mhlo"
    "-iree-mlir-to-vm-bytecode-module"
    "-iree-hal-target-backends=dylib-llvm-aot"
    "-iree-llvm-static-library-output-path=${CMAKE_CURRENT_BINARY_DIR}/simple_mul.o"
  PUBLIC
  TESTONLY
  GENERATED
    ${CMAKE_CURRENT_BINARY_DIR}/simple_mul.o
  )

  if(DEFINED _RULE_TRANSLATE_TOOL)
    set(_TRANSLATE_TOOL ${_RULE_TRANSLATE_TOOL})
  else()
    set(_TRANSLATE_TOOL "iree-translate")
  endif()

  iree_get_executable_path(_TRANSLATE_TOOL_EXECUTABLE ${_TRANSLATE_TOOL})

  set(_ARGS)
  list(APPEND _ARGS "-iree-input-type=mhlo"  )
  list(APPEND _ARGS "-iree-mlir-to-vm-bytecode-module")
  list(APPEND _ARGS "-iree-hal-target-backends=dylib-llvm-aot")
  list(APPEND _ARGS "-iree-llvm-static-library-output-path=simple_mul.o")
  list(APPEND _ARGS "${CMAKE_CURRENT_SOURCE_DIR}/simple_mul.mlir")
  list(APPEND _ARGS "-o")
  list(APPEND _ARGS "simple_mul.vmfb")

  # Depending on the binary instead of the target here given we might not have
  # a target in this CMake invocation when cross-compiling.
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/simple_mul.o
    COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${_ARGS}
    # Changes to either the translation tool or the input source should
    # trigger rebuilding.
    DEPENDS ${_TRANSLATE_TOOL_EXECUTABLE} "simple_mul.mlir"
  )

  add_library(simple_mul
        STATIC
        ${CMAKE_CURRENT_BINARY_DIR}/simple_mul.o)

  SET_TARGET_PROPERTIES(
    simple_mul
    PROPERTIES
    LINKER_LANGUAGE C
    )

iree_cc_binary(
  NAME
    static_library_demo
  SRCS
    "static_library_demo.c"
  DEPS
    ::simple_mul_c
    iree::runtime
    iree::base
    iree::hal
    iree::hal::local
    iree::hal::local::loaders::static_library_loader
    iree::hal::local::task_driver
    iree::modules::hal
    iree::task::api
    iree::vm
    iree::vm::bytecode_module
    simple_mul
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
