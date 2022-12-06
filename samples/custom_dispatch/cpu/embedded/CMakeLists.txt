# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_LLVM_CPU OR
   NOT IREE_HAL_DRIVER_LOCAL_SYNC OR
   NOT IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
  return()
endif()

# NOTE: this example uses clang to target only aarch64 and x86_64 as that's what
# the example.mlir file is hardcoded to accept. It's possible to use arbitrary
# compilers when using either the system linker or static library output options
# of the CPU backend but embedded dynamic libraries have strict requirements and
# it's easiest to always use clang (plus it is good at cross-compiling) and the
# settings below.
#
# This example just shows how users can link in custom objects and is not
# intended to demonstrate the infrastructure to produce the object files: when
# using this custom kernel approach it is up to the user to handle that work.
find_program(CLANG clang)
if(NOT CLANG)
  message(STATUS "IREE custom_dispatch/cpu/embedded ignored -- clang not found")
  return()
endif()

# This only builds for x86-64 because this is just a sample and we don't feature
# detect what backends are compiled into clang. We could extend this to build
# for the current cmake target architecture but would also need to modify the
# MLIR file to have the new target configuration.
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)")
  message(STATUS "IREE custom_dispatch/cpu/embedded ignored -- only builds for x86_64 (today)")
  return()
endif()

function(embedded_function_object _ARCH)
  set(_NAME iree_samples_custom_dispatch_cpu_embedded_object_${_ARCH})
  add_custom_command(
    OUTPUT functions_${_ARCH}.o
    DEPENDS functions.c
    COMMAND ${CLANG}
        -target ${_ARCH}-unknown-unknown-eabi-elf
        -std=c17
        -fvisibility=hidden
        -fno-plt
        -fno-rtti
        -fno-exceptions
        -fdata-sections
        -ffunction-sections
        -funique-section-names
        -c ${CMAKE_CURRENT_SOURCE_DIR}/functions.c
        -o ${CMAKE_CURRENT_BINARY_DIR}/functions_${_ARCH}.o
    VERBATIM
  )
  add_custom_target(${_NAME} DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/functions_${_ARCH}.o
  )
  add_dependencies(iree-sample-deps "${_NAME}")
endfunction()

# Build the functions_*.o files for each architecture we target.
embedded_function_object(aarch64)
embedded_function_object(x86_64)

iree_lit_test_suite(
  NAME
    examples
  SRCS
    "example_hal.mlir"
    "example_stream.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
  LABELS
    "driver=local-sync"
    "hostonly"
)
