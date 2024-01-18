# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_HAL_DRIVER_LOCAL_SYNC OR
   NOT IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
  return()
endif()

# This only builds for x86-64 and arm64. We could extend this to build
# for the current cmake target architecture but would also need to modify the
# MLIR file to have the new target configuration.
iree_compiler_targeting_iree_arch(_BUILD_COMPILER_TARGETING_X86_64 "x86_64")
iree_compiler_targeting_iree_arch(_BUILD_COMPILER_TARGETING_ARM_64 "arm_64")
# First check compiler support
if(NOT _BUILD_COMPILER_TARGETING_X86_64 OR NOT _BUILD_COMPILER_TARGETING_ARM_64)
  message(STATUS "IREE custom_dispatch/cpu/embedded ignored - some required LLVM targets are not built.")
  return()
endif()
# Check if the host architecture matches what we are building code for.
if(NOT IREE_ARCH STREQUAL "x86_64" AND NOT IREE_ARCH STREQUAL "arm_64")
  message(STATUS "IREE custom_dispatch/cpu/embedded ignored - host architecture unsupported.")
  return()
endif()

function(embedded_function_object _ARCH)
  set(_NAME iree_samples_custom_dispatch_cpu_embedded_object_${_ARCH})
  iree_arch_to_llvm_arch(_LLVM_ARCH ${_ARCH})
  add_custom_command(
    OUTPUT
      functions_${_ARCH}.o
    DEPENDS
      functions.c
      ${IREE_CLANG_TARGET}
    COMMAND ${IREE_CLANG_BINARY}
      -target ${_LLVM_ARCH}-none-elf
      -isystem ${IREE_CLANG_BUILTIN_HEADERS_PATH}
      -std=c17
      -ffreestanding
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
embedded_function_object(arm_64)
embedded_function_object(x86_64)

iree_lit_test_suite(
  NAME
    examples
  SRCS
    "example_hal.mlir"
    "example_stream.mlir"
    "example_udon.mlir"
  DATA
  functions_arm_64.o
  functions_x86_64.o
  udon_spec.mlir
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
  LABELS
    "driver=local-sync"
    "hostonly"
)
