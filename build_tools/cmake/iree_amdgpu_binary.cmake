# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# Builds an LLVM shared library for AMDGPU from input files via clang.
#
# Parameters:
# NAME: Name of the target.
# OUT: Output file name.
# TARGET: LLVM `-target` flag.
# ARCH: LLVM `-march` flag.
# SRCS: source files to pass to clang.
# INTERNAL_HDRS: all headers transitively included by the source files.
#                Unlike typical Bazel `hdrs`, these are not exposed as
#                interface headers. This would normally be part of `srcs`,
#                but separating it was easier for `bazel_to_cmake`, as
#                CMake does not need this, and making this explicitly
#                Bazel-only allows using `filegroup` on the Bazel side.
# COPTS: additional flags to pass to clang.
# LINKOPTS: additional flags to pass to lld.
function(iree_amdgpu_binary)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT;TARGET;ARCH"
    "SRCS;INTERNAL_HDRS;COPTS;LINKOPTS"
    ${ARGN}
  )

  iree_package_name(_PACKAGE_NAME)

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}.so")
  endif()

  set(_COPTS
    # C configuration.
    "-x" "c"
    "-Xclang" "-finclude-default-header"
    "-std=c23"
    "-nogpulib"
    "-fno-short-wchar"

    # Target architecture/machine.
    "-target" "${_RULE_TARGET}"
    "-march=${_RULE_ARCH}"
    "-fgpu-rdc"  # NOTE: may not be required for all targets

    # Header paths for builtins and our own includes.
    "-isystem" "${IREE_CLANG_BUILTIN_HEADERS_PATH}"
    "-I${IREE_SOURCE_DIR}/runtime/src"
    "-I${IREE_BINARY_DIR}/runtime/src"

    # Avoid warnings about things we do that are not compatible across compilers
    # but are fine because we're only ever compiling with clang.
    "-Wno-gnu-pointer-arith"

    # Optimized.
    "-fno-ident"
    "-fvisibility=hidden"
    "-O3"

    # Object file only in bitcode format.
    "-c"
    "-emit-llvm"
  )

  set(_BITCODE_FILES)
  foreach(_SRC ${_RULE_SRCS})
    get_filename_component(_BITCODE_SRC_PATH "${_SRC}" REALPATH)
    string(REGEX REPLACE "[.]c$" ".bc" _BITCODE_FILE ${_SRC})
    list(APPEND _BITCODE_FILES ${_BITCODE_FILE})
    add_custom_command(
      OUTPUT
        "${_BITCODE_FILE}"
      COMMAND
        "${IREE_CLANG_BINARY}"
        ${_COPTS}
        "${_BITCODE_SRC_PATH}"
        "-o"
        "${_BITCODE_FILE}"
      DEPENDS
        "${IREE_CLANG_BINARY}"
        "${_BITCODE_SRC_PATH}"
        "${_RULE_INTERNAL_HDRS}"
      MAIN_DEPENDENCY
        "${_BITCODE_SRC_PATH}"
      COMMENT
        "Compiling ${_SRC} to ${_BITCODE_FILE}"
      VERBATIM
    )
  endforeach()

  set(_ARCHIVE_FILE "${_RULE_NAME}.a")
  add_custom_command(
    OUTPUT
      ${_ARCHIVE_FILE}
    COMMAND
      ${IREE_LLVM_LINK_BINARY}
      ${_BITCODE_FILES}
      "-o"
      "${_ARCHIVE_FILE}"
    DEPENDS
      ${IREE_LLVM_LINK_BINARY}
      ${_BITCODE_FILES}
    COMMENT
      "Archiving bitcode to ${_ARCHIVE_FILE}"
    VERBATIM
  )

  set(_LINKED_FILE "${_RULE_NAME}.bc")
  add_custom_command(
    OUTPUT
      ${_LINKED_FILE}
    COMMAND
      ${IREE_LLVM_LINK_BINARY}
      "-internalize"
      "-only-needed"
      "${_ARCHIVE_FILE}"
      "-o" "${_LINKED_FILE}"
    DEPENDS
      "${IREE_LLVM_LINK_BINARY}"
      "${_ARCHIVE_FILE}"
    COMMENT
      "Linking bitcode to ${_LINKED_FILE}"
    VERBATIM
  )

  add_custom_command(
    OUTPUT
      "${_OUT}"
    COMMAND
      ${IREE_LLD_BINARY}
      "-flavor" "gnu"
      "-m" "elf64_amdgpu"
      "--build-id=none"
      "--no-undefined"
      "-shared"
      "-plugin-opt=mcpu=${_RULE_ARCH}"
      "-plugin-opt=O3"
      "--lto-CGO3"
      "--no-whole-archive"
      "--gc-sections"
      "--strip-debug"
      "--discard-all"
      "--discard-locals"
      "${_LINKED_FILE}"
      "-o" "${_OUT}"
    DEPENDS
      "${_LINKED_FILE}"
      "${IREE_LLD_TARGET}"
    COMMENT
      "Compiling binary to ${_OUT}"
    VERBATIM
  )

  # Only add iree_${NAME} as custom target doesn't support aliasing to
  # iree::${NAME}.
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_OUT}"
  )
endfunction()
