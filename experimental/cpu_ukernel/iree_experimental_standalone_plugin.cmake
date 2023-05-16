# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_experimental_standalone_plugin_arch()
#
# Helper for iree_experimental_standalone_plugin, building one
# architecture.
#
# Parameters:
# NAME: Name of the system plugin to create.
# ARCH: Name of architecture (as in IREE_ARCH) to build for.
#       Example: "arm_64".
# COPTS: List of compiler options to be applied to all source files.
# SRCS: List of source files. Each list entry may be of one of two forms:
#         * Each entry that does not contain a colon is interpreted as a source
#           file path, to be built unconditionally, with the compiler options
#           specified in `COPTS`.
#         * Each entry that contains a colon is interpreted as a colon-separated
#           list of length either 2 or 3. Format:
#           `ARCH:FILE[:FILE_COPTS_VAR_NAME]`.
#           Any entry whose `ARCH` does not match this rules's `ARCH` parameter
#           is filtered out. Remaining files are compiled with the
#           architecture-wide compiler options (see `COPTS`) and, if provided,
#           with the file-specific compiler options from expanding the variable
#           specified in `FILE_COPTS_VAR_NAME`.
#           Example:  "x86_64:some_file_for_x86_64_using_avx512_instructions.c:NAME_OF_VARIABLE_CONTAINING_COPTS_FOR_X86_64_AVX512".
function(iree_experimental_standalone_plugin_arch)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;ARCH"
    "SRCS;COPTS"
    ${ARGN}
  )

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}_${_RULE_ARCH}")
  iree_arch_to_llvm_arch(LLVM_ARCH "${_RULE_ARCH}")

  foreach(_SRC_ENTRY_COLON_SEPARATED IN LISTS _RULE_SRCS)
    string(REPLACE ":" ";" _SRC_ENTRY_LIST "${_SRC_ENTRY_COLON_SEPARATED}")
    list(LENGTH _SRC_ENTRY_LIST _SRC_ENTRY_LIST_LENGTH)
    set(_SRC_COPTS_VAR_NAME "")
    set(_SRC_FILE "")
    if(_SRC_ENTRY_LIST_LENGTH EQUAL 1)
      set(_SRC_FILE "${_SRC_ENTRY_LIST}")
    else()  # NOT _SRC_ENTRY_LIST_LENGTH EQUAL 1
      list(GET _SRC_ENTRY_LIST 0 _SRC_ARCH)
      if(NOT _SRC_ARCH STREQUAL _RULE_ARCH)
        continue()
      endif()
      list(GET _SRC_ENTRY_LIST 1 _SRC_FILE)
      if(_SRC_ENTRY_LIST_LENGTH EQUAL 3)
        list(GET _SRC_ENTRY_LIST 2 _SRC_COPTS_VAR_NAME)
      endif()
    endif()  # NOT _SRC_ENTRY_LIST_LENGTH EQUAL 1

    set(_SRC_COPTS "${${_SRC_COPTS_VAR_NAME}}")

    get_filename_component(_SRC_FILE_BASENAME "${_SRC_FILE}" NAME)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_SRC_FILE}")
      set(_SRC_FILE "${CMAKE_CURRENT_SOURCE_DIR}/${_SRC_FILE}")
    endif()

    if(EXISTS "${PROJECT_SOURCE_DIR}/${_SRC_FILE}")
      set(_SRC_FILE "${PROJECT_SOURCE_DIR}/${_SRC_FILE}")
    endif()

    set(_OBJECT_FILE "${_SRC_FILE_BASENAME}.${_RULE_ARCH}.o")
    list(APPEND _OBJECT_FILES "${CMAKE_CURRENT_BINARY_DIR}/${_OBJECT_FILE}")
    add_custom_command(
      OUTPUT
        "${_OBJECT_FILE}"
      DEPENDS
        "${_SRC_FILE}"
        "${IREE_CLANG_TARGET}"
      COMMAND "${IREE_CLANG_TARGET}"
        # Flags copied from
        # compiler/src/iree/compiler/Dialect/HAL/Target/LLVMCPU/internal/EmbeddedLinkerTool.cpp
        -target "${LLVM_ARCH}-unknown-unknown-eabi-elf"
        -isystem "${IREE_BINARY_DIR}/third_party/llvm-project/llvm/lib/clang/17/include"
        -std=c17
        -fasm  # Added for inline-asm support.
        -fPIC
        -ffreestanding
        -fvisibility=hidden
        -fno-plt
        -fno-rtti
        -fno-exceptions
        -fdata-sections
        -ffunction-sections
        -funique-section-names
        -DIREE_UK_STANDALONE
        -I "${IREE_SOURCE_DIR}/runtime/src/"
        -c "${_SRC_FILE}"
        -o "${CMAKE_CURRENT_BINARY_DIR}/${_OBJECT_FILE}"
        ${_RULE_COPTS}
        ${_SRC_COPTS}
      VERBATIM
    )
  endforeach()
  set(_OUTPUT_SO_FILE "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.${_RULE_ARCH}.so")
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

# iree_experimental_standalone_plugin()
#
# Creates a standalone plugin library, that is built using our in-tree Clang
# toolchain for multiple target architectures, generating a fat embedded-elf,
# and may be loaded with the embedded dynamic library loaded. 
#
# Contrast with: iree_experimental_system_plugin.
#
# Parameters:
# NAME: Name of the system plugin to create.
# ARCHS: List of architectures (as in IREE_ARCH) to build. Format:
#        `ARCH[:ARCH_COPTS_VAR_NAME]`. If provided, `ARCH_COPTS_VAR_NAME` is
#        interpreted as the name of a variable to be expanded into all compiler
#        command lines used for architecture `ARCH`.
#        Example: "arm_64:NAME_OF_VARIABLE_CONTAINING_COPTS_FOR_ARM_64".
# SRCS: List of source files. Each list entry may be of one of two forms:
#         * Each entry that does not contain a colon is interpreted as a source
#           file path, to be built for all architectures with the
#           architecture-wide compiler options provided for each architecture
#           (see `ARCHS`).
#         * Each entry that contains a colon is interpreted as a colon-separated
#           list of length either 2 or 3. Format:
#           `ARCH:FILE[:FILE_COPTS_VAR_NAME]`.
#           The specified source `FILE` is compiled only for the specified
#           architecture `ARCH` and is skipped on other architectures. It is
#           compiled with the architecture-wide compiler options
#           (see `ARCHS`) and, if provided, with the file-specific compiler
#           options from expanding the variable specified in
#           `FILE_COPTS_VAR_NAME`.
#           Example:  "x86_64:some_file_for_x86_64_using_avx512_instructions.c:NAME_OF_VARIABLE_CONTAINING_COPTS_FOR_X86_64_AVX512".
function(iree_experimental_standalone_plugin)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;ARCHS"
    ${ARGN}
  )

  # Iterate over architectures. For each of them, build the architecture-specific
  # shared library (iree_experimental_standalone_plugin_arch).
  foreach(_ARCH_ENTRY_COLON_SEPARATED IN LISTS _RULE_ARCHS)
    # Turn the colon-separated ARCH entry into a CMake list (semicolon-separated)
    string(REPLACE ":" ";" _ARCH_ENTRY_LIST "${_ARCH_ENTRY_COLON_SEPARATED}")
    list(GET _ARCH_ENTRY_LIST 0 _ARCH)
    list(LENGTH _ARCH_ENTRY_LIST _ARCH_ENTRY_LIST_LENGTH)
    # Get optional architecture-wide copts into _COPTS.
    set(_COPTS_VAR_NAME "")
    if(_ARCH_ENTRY_LIST_LENGTH EQUAL 2)
      list(GET _ARCH_ENTRY_LIST 1 _COPTS_VAR_NAME)
    endif()
    set(_COPTS "${${_COPTS_VAR_NAME}}")
    # Build the architecture-specific shared library.
    iree_experimental_standalone_plugin_arch(
      NAME
        "${_RULE_NAME}"
      ARCH
        "${_ARCH}"
      SRCS
        ${_RULE_SRCS}
      COPTS
        ${_COPTS}
    )
    list(APPEND _ARCH_SO_FILES "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_NAME}.${_ARCH}.so")
  endforeach()
  # Generate the multi-architecture ELF file.
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
  add_dependencies(iree-test-deps "${_NAME}")
endfunction()
