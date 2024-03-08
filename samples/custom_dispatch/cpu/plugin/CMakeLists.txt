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

add_dependencies(iree-sample-deps
  iree_samples_custom_dispatch_cpu_system_plugin)

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
# compiler/plugins/target/LLVMCPU/internal/EmbeddedLinkerTool.cpp
if(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF)

# This only builds for x86-64/arm_64/riscv_64 because that's all we have coded in here.
if(NOT IREE_ARCH STREQUAL "arm_64" AND NOT IREE_ARCH STREQUAL "x86_64" AND NOT IREE_ARCH STREQUAL "riscv_64")
  message(STATUS "IREE custom_dispatch/cpu/plugin standalone example ignored -- only builds for x86_64/arm_64/riscv_64 (today)")
  return()
endif()

# TODO(#12801): fix MSVC embedded elf -> embedded elf calling.
if(MSVC)
  message(STATUS "IREE custom_dispatch/cpu/plugin standalone example ignored -- #12801 required to make MSVC work")
  return()
endif()

# standalone_plugin_library()
#
# Creates the architecture-specific part of the standalone plugin, for the given
# architecture (as in IREE_ARCH), and appends the generated library filename to
# the output-variable _DST_VAR_ARCH_LIBRARIES. Skips if the compiler does not
# support the specified architecture.
function(standalone_plugin_library _DST_VAR_ARCH_LIBRARIES _ARCH _CFLAGS)
  iree_compiler_targeting_iree_arch(_ENABLED "${_ARCH}")
  if(NOT _ENABLED)
    return()
  endif()
  set(_NAME iree_samples_custom_dispatch_cpu_standalone_plugin_${_ARCH})
  iree_arch_to_llvm_arch(LLVM_ARCH "${_ARCH}")
  add_custom_command(
    OUTPUT
      standalone_plugin_${_ARCH}.o
    DEPENDS
      standalone_plugin.c
      ${IREE_CLANG_TARGET}
    COMMAND ${IREE_CLANG_BINARY}
      -target ${LLVM_ARCH}-none-elf
      -isystem ${IREE_CLANG_BUILTIN_HEADERS_PATH}
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
      ${_CFLAGS}
      -I ${IREE_SOURCE_DIR}/runtime/src/
      -c ${CMAKE_CURRENT_SOURCE_DIR}/standalone_plugin.c
      -o ${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_${_ARCH}.o
    VERBATIM
  )
  set(_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/standalone_plugin_${_ARCH}.so")
  add_custom_command(
    OUTPUT
      standalone_plugin_${_ARCH}.so
    DEPENDS
      standalone_plugin_${_ARCH}.o
      ${IREE_LLD_TARGET}
    COMMAND ${IREE_LLD_BINARY}
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
      -o "${_LIBRARY}"
    VERBATIM
  )
  add_custom_target(${_NAME} DEPENDS
    "${_LIBRARY}"
  )
  set(_LOCAL_ARCH_LIBRARIES "${${_DST_VAR_ARCH_LIBRARIES}}")
  list(APPEND _LOCAL_ARCH_LIBRARIES "${_LIBRARY}")
  set("${_DST_VAR_ARCH_LIBRARIES}" "${_LOCAL_ARCH_LIBRARIES}" PARENT_SCOPE)
endfunction()

# Build the standalone_plugin_*.so files for each architecture we target.
set(_STANDALONE_PLUGIN_ARCH_LIBRARIES "")


set(RISCV_STANDALONE_PLUGIN_ARCH_C_FLAGS -march=rv64i2p1ma2p1f2p2d2p2c2p0)
standalone_plugin_library(_STANDALONE_PLUGIN_ARCH_LIBRARIES arm_64 "")
standalone_plugin_library(_STANDALONE_PLUGIN_ARCH_LIBRARIES x86_64 "")
standalone_plugin_library(_STANDALONE_PLUGIN_ARCH_LIBRARIES riscv_64 "${RISCV_STANDALONE_PLUGIN_ARCH_C_FLAGS}")

add_custom_command(
  OUTPUT
    standalone_plugin.sos
  DEPENDS
    ${_STANDALONE_PLUGIN_ARCH_LIBRARIES}
    iree-fatelf
  COMMAND iree-fatelf join
    ${_STANDALONE_PLUGIN_ARCH_LIBRARIES}
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
