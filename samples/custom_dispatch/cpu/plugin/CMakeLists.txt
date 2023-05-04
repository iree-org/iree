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

# System plugins using iree_samples_system_plugin_cc_library.
if(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY)

# Creates a system plugin library, that is built using the host toolchain for
# the host architecture and may be loaded with the system dynamic library
# loader.
#
# Contrast with: iree_samples_standalone_plugin_cc_library.
function(iree_samples_system_plugin_cc_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DEPS"
    ${ARGN}
  )

  iree_cc_library(
    NAME
      ${_RULE_NAME}
    SRCS
      ${_RULE_SRCS}
    DEPS
      ${_RULE_DEPS}
      iree::hal::local::executable_plugin
    INCLUDES
      "${IREE_SOURCE_DIR}/runtime/src/"
    SHARED
  )

  # NOTE: this is only required because we want this sample to run on all
  # platforms without needing to change the library name (libfoo.so/foo.dll).
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  set_target_properties("${_NAME}"
    PROPERTIES
      WINDOWS_EXPORT_ALL_SYMBOLS ON
      PREFIX ""
      OUTPUT_NAME "${_RULE_NAME}"
  )

  add_dependencies(iree-sample-deps "${_NAME}")
endfunction()

iree_samples_system_plugin_cc_library(
  NAME
    simple_mul_workgroup_system_plugin
  SRCS
    simple_mul_workgroup_system_plugin.c
)

iree_samples_system_plugin_cc_library(
  NAME
    builtin_ukernel_system_plugin
  SRCS
    builtin_ukernel_system_plugin.c
  DEPS
    iree::builtins::ukernel
)

iree_lit_test_suite(
  NAME
    simple_mul_workgroup_system_example
  SRCS
    "simple_mul_workgroup_system_example.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
    iree_samples_custom_dispatch_cpu_plugin_simple_mul_workgroup_system_plugin
  LABELS
    "driver=local-sync"
    "hostonly"
)

iree_lit_test_suite(
  NAME
    simple_mul_workgroup_system_ukernel
  SRCS
    "simple_mul_workgroup_system_ukernel.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
    iree_samples_custom_dispatch_cpu_plugin_simple_mul_workgroup_system_plugin
  LABELS
    "driver=local-sync"
    "hostonly"
)

iree_lit_test_suite(
  NAME
    builtin_ukernel_system_plugin_test
  SRCS
    "builtin_ukernel_system_plugin_test.mlir"
  TOOLS
    iree-compile
    iree-run-module
    iree_samples_custom_dispatch_cpu_plugin_builtin_ukernel_system_plugin
  LABELS
    "driver=local-sync"
    "hostonly"
)

endif(IREE_HAL_EXECUTABLE_PLUGIN_SYSTEM_LIBRARY)

# embedded-elf plugin mechanism for standalone portable ELFs.
# We use the same flags we do in the compiler for our embedded elfs found under
# compiler/src/iree/compiler/Dialect/HAL/Target/LLVMCPU/internal/EmbeddedLinkerTool.cpp
if(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF)

# This only builds for x86-64/arm_64 because that's all we have coded in here.
if(NOT IREE_ARCH STREQUAL "arm_64" AND NOT IREE_ARCH STREQUAL "x86_64")
  message(STATUS "IREE custom_dispatch/cpu/plugin standalone example ignored -- only builds for x86_64/arm_64 (today)")
  return()
endif()

# TODO(#12801): fix MSVC embedded elf -> embedded elf calling.
if(MSVC)
  message(STATUS "IREE custom_dispatch/cpu/plugin standalone example ignored -- #12801 required to make MSVC work")
  return()
endif()

# Helper for iree_samples_standalone_plugin_cc_library, building one
# architecture.
function(iree_samples_standalone_plugin_cc_library_arch)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;ARCH"
    "SRCS"
    ${ARGN}
  )

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}_${_RULE_ARCH}")

  if (_RULE_ARCH STREQUAL "arm_64")
    set(LLVM_ARCH "aarch64")
  else()
    set(LLVM_ARCH "${_RULE_ARCH}")
  endif()

  foreach(_SRC_FILE IN LISTS _RULE_SRCS)
    set(_OBJECT_FILE "${_SRC_FILE}_${_RULE_ARCH}.o")
    list(APPEND _OBJECT_FILES "${CMAKE_CURRENT_BINARY_DIR}/${_OBJECT_FILE}")
    add_custom_command(
      OUTPUT
        "${_OBJECT_FILE}"
      DEPENDS
        "${_SRC_FILE}"
        "${IREE_CLANG_TARGET}"
      COMMAND "${IREE_CLANG_TARGET}"
        -target "${LLVM_ARCH}-unknown-unknown-eabi-elf"
        -isystem "${IREE_BINARY_DIR}/third_party/llvm-project/llvm/lib/clang/17/include"
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
        -I "${IREE_SOURCE_DIR}/runtime/src/"
        -c "${CMAKE_CURRENT_SOURCE_DIR}/${_SRC_FILE}"
        -o "${CMAKE_CURRENT_BINARY_DIR}/${_OBJECT_FILE}"
      VERBATIM
    )
  endforeach()
  set (_OUTPUT_SO_FILE "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_${_RULE_ARCH}.so")
  add_custom_command(
    OUTPUT
      ${_OUTPUT_SO_FILE}
    DEPENDS
      ${_OBJECT_FILES}
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
      ${_OBJECT_FILES}
      -o "${_OUTPUT_SO_FILE}"
    VERBATIM
  )
  add_custom_target(${_NAME} DEPENDS
    "${_OUTPUT_SO_FILE}"
  )
endfunction()

# Creates a standalone plugin library, that is built using our in-tree Clang
# toolchain for multiple target architectures, generating a fat embedded-elf,
# and may be loaded with the embedded dynamic library loaded. 
#
# Contrast with: iree_samples_system_plugin_cc_library.
function(iree_samples_standalone_plugin_cc_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;ARCHS"
    ${ARGN}
  )

  foreach(_ARCH IN LISTS _RULE_ARCHS)
    iree_samples_standalone_plugin_cc_library_arch(
      NAME
        "${_RULE_NAME}"
      ARCH
        "${_ARCH}"
      SRCS
        "${_RULE_SRCS}"
    )
    list(APPEND _ARCH_SO_FILES "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}_${_ARCH}.so")
  endforeach()

  add_custom_command(
    OUTPUT
      "${_RULE_NAME}.sos"
    DEPENDS
      ${_ARCH_SO_FILES}
      iree-fatelf
    COMMAND iree-fatelf join
      ${_ARCH_SO_FILES}
      > ${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.sos
    VERBATIM
  )
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
  add_custom_target("${_NAME}" DEPENDS
    "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.sos"
  )
  add_dependencies(iree-sample-deps "${_NAME}")
endfunction()

iree_samples_standalone_plugin_cc_library(
  NAME
    simple_mul_workgroup_standalone_plugin
  ARCHS
    arm_64
    x86_64
  SRCS
    simple_mul_workgroup_standalone_plugin.c
)

iree_lit_test_suite(
  NAME
    simple_mul_workgroup_standalone_example
  SRCS
    "simple_mul_workgroup_standalone_example.mlir"
  DATA
    simple_mul_workgroup_standalone_plugin.sos
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
  simple_mul_workgroup_standalone_ukernel
  SRCS
    "simple_mul_workgroup_standalone_ukernel.mlir"
  DATA
    simple_mul_workgroup_standalone_plugin.sos
  TOOLS
    FileCheck
    iree-compile
    iree-run-module
  LABELS
    "driver=local-sync"
    "hostonly"
)

endif(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF)
