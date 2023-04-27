# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT IREE_TARGET_BACKEND_LLVM_CPU OR
   NOT IREE_HAL_DRIVER_LOCAL_SYNC OR
   NOT IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF)
  return()
endif()

# system-library plugin mechanism using the system dynamic library loader.
if(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY)

add_library(iree_samples_custom_dispatch_cpu_system_plugin SHARED
  system_plugin.c
)
target_include_directories(iree_samples_custom_dispatch_cpu_system_plugin
  PRIVATE
    ${IREE_SOURCE_DIR}/runtime/src/
)

# NOTE: this is only required because we want this sample to run on all
# platforms without needing to change the library name (libfoo.so/foo.dll).
set_target_properties(iree_samples_custom_dispatch_cpu_system_plugin
  PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    PREFIX ""
    OUTPUT_NAME "system_plugin"
)

add_dependencies(iree-sample-deps iree_samples_custom_dispatch_cpu_system_plugin)

iree_lit_test_suite(
  NAME
    system_example
  SRCS
    "system_example.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
    iree_samples_custom_dispatch_cpu_system_plugin
  LABELS
    "driver=local-sync"
    "hostonly"
)

iree_lit_test_suite(
  NAME
    system_ukernel
  SRCS
    "system_ukernel.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
    iree_samples_custom_dispatch_cpu_system_plugin
  LABELS
    "driver=local-sync"
    "hostonly"
)

endif(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY)

# embedded-elf plugin mechanism for standalone portable ELFs.
# We use the same flags we do in the compiler for our embedded elfs found under
# compiler/src/iree/compiler/Dialect/HAL/Target/LLVMCPU/internal/EmbeddedLinkerTool.cpp
if(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF)

# This only builds for x86-64/aarch64 because that's all we have coded in here.
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)" AND
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
  message(STATUS "IREE custom_dispatch/cpu/plugin standalone example ignored -- only builds for x86_64/aarch64 (today)")
  return()
endif()

# TODO(#12801): fix MSVC embedded elf -> embedded elf calling.
if(MSVC)
  message(STATUS "IREE custom_dispatch/cpu/plugin standalone example ignored -- #12801 required to make MSVC work")
  return()
endif()

function(standalone_plugin_library _ARCH)
  set(_NAME iree_samples_custom_dispatch_cpu_standalone_plugin_${_ARCH})
  add_custom_command(
    OUTPUT
      standalone_plugin_${_ARCH}.o
    DEPENDS
      standalone_plugin.c
      ${IREE_CLANG_TARGET}
    COMMAND ${IREE_CLANG_TARGET}
      -target ${_ARCH}-unknown-unknown-eabi-elf
      -isystem ${IREE_BINARY_DIR}/third_party/llvm-project/llvm/lib/clang/17/include
      -std=c17
      -fPIC
      -ffreestanding
      -fvisibility=hidden
      -fno-plt
      -fno-rtti
      -fno-exceptions
      -fdata-sections
      -ffunction-sections
      -funique-section-names
      -I ${IREE_SOURCE_DIR}/runtime/src/
      -c ${CMAKE_CURRENT_SOURCE_DIR}/standalone_plugin.c
      -o ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_${_ARCH}.o
    VERBATIM
  )
  add_custom_command(
    OUTPUT
      standalone_plugin_${_ARCH}.so
    DEPENDS
      standalone_plugin_${_ARCH}.o
      ${IREE_LLD_TARGET}
    COMMAND ${IREE_LLD_TARGET}
      -flavor gnu
      --build-id=none
      -nostdlib
      -static
      -shared
      --no-undefined
      --no-allow-shlib-undefined
      --allow-multiple-definition
      --gc-sections
      -z now
      -z relro
      --discard-all
      --icf=all
      --ignore-data-address-equality
      --ignore-function-address-equality
      --hash-style=sysv
      --strip-debug
      ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_${_ARCH}.o
      -o ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_${_ARCH}.so
    VERBATIM
  )
  add_custom_target(${_NAME} DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_${_ARCH}.so
  )
endfunction()

# Build the standalone_plugin_*.so files for each architecture we target.
standalone_plugin_library(aarch64)
standalone_plugin_library(x86_64)

add_custom_command(
  OUTPUT
    standalone_plugin.sos
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_aarch64.so
    ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_x86_64.so
    iree-fatelf
  COMMAND iree-fatelf join
    ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_aarch64.so
    ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_x86_64.so
    > ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin.sos
  VERBATIM
)
add_custom_target(iree_samples_custom_dispatch_cpu_standalone_plugin DEPENDS
  ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin.sos
)

add_dependencies(iree-sample-deps iree_samples_custom_dispatch_cpu_standalone_plugin)

iree_lit_test_suite(
  NAME
    standalone_example
  SRCS
    "standalone_example.mlir"
  DATA
    standalone_plugin.sos
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
  LABELS
    "driver=local-sync"
    "hostonly"
)

iree_lit_test_suite(
  NAME
    standalone_ukernel
  SRCS
    "standalone_ukernel.mlir"
  DATA
    standalone_plugin.sos
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
  LABELS
    "driver=local-sync"
    "hostonly"
)

endif(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF)
