# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_bitcode_library()
#
# Builds an LLVM bitcode library from an input file via clang
#
# Parameters:
# NAME: Name of target (see Note).
# SRCS: Source files. Headers go here as well, as in iree_cc_library. There is
#       no concept of public headers (HDRS) here.
# COPTS: additional flags to pass to clang.
# OUT: Output file name (defaults to NAME.bc).
function(iree_bitcode_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT;ARCH"
    "SRCS;COPTS"
    ${ARGN}
  )

  set(_CLANG_TOOL "$<TARGET_FILE:${IREE_CLANG_TARGET}>")
  set(_LINK_TOOL "$<TARGET_FILE:${IREE_LLVM_LINK_TARGET}>")

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}.bc")
  endif()

  # We need CLANG_VERSION_MAJOR to set up include directories. Unfortunately,
  # Clang's own CMakeLists do not expose CLANG_VERSION_MAJOR to PARENT_SCOPE.
  # Likewise with LLVM_VERSION_MAJOR. However, CLANG_EXECUTABLE_VERSION is
  # CACHE'd, so we can access it, and it currently has the same value.
  set(_CLANG_VERSION_MAJOR "${CLANG_EXECUTABLE_VERSION}")

  # These are copied as part of the clang build; we could allow the user to
  # override this but it should be harmless.
  set(_BUILTIN_HEADERS_PATH "${IREE_BINARY_DIR}/llvm-project/lib/clang/${_CLANG_VERSION_MAJOR}/include/")

  iree_arch_to_llvm_arch(_LLVM_ARCH "${_RULE_ARCH}")

  set(_COPTS
    # Target architecture.
    "-target" "${_LLVM_ARCH}"

    # C17 with no system deps.
    "-std=c17"
    "-nostdinc"
    "-ffreestanding"

    # Optimized and unstamped.
    "-O3"
    "-DNDEBUG"
    "-fno-ident"
    "-fdiscard-value-names"

    # Set the size of wchar_t to 4 bytes (instead of 2 bytes).
    # This must match what the runtime is built with.
    "-fno-short-wchar"

    # Object file only in bitcode format:
    "-c"
    "-emit-llvm"

    # Force the library into standalone mode (not depending on build-directory
    # configuration).
    "-DIREE_DEVICE_STANDALONE=1"
  )

  list(APPEND _COPTS "-isystem" "${_BUILTIN_HEADERS_PATH}")
  list(APPEND _COPTS "-I" "${IREE_SOURCE_DIR}/runtime/src")
  list(APPEND _COPTS "-I" "${IREE_BINARY_DIR}/runtime/src")
  list(APPEND _COPTS "${_RULE_COPTS}")

  set(_BITCODE_FILES)
  foreach(_SRC ${_RULE_SRCS})
    get_filename_component(_BITCODE_SRC_PATH "${_SRC}" REALPATH)
    set(_BITCODE_FILE "${_RULE_NAME}_${_SRC}.bc")
    list(APPEND _BITCODE_FILES ${_BITCODE_FILE})
    add_custom_command(
      OUTPUT
        "${_BITCODE_FILE}"
      COMMAND
        "${_CLANG_TOOL}"
        ${_COPTS}
        "${_BITCODE_SRC_PATH}"
        "-o"
        "${_BITCODE_FILE}"
      DEPENDS
        "${_CLANG_TOOL}"
        "${_LINK_TOOL}"
        "${_SRC}"
      COMMENT
        "Compiling ${_SRC} to ${_BITCODE_FILE}"
      VERBATIM
    )
  endforeach()

  add_custom_command(
    OUTPUT
      ${_OUT}
    COMMAND
      ${_LINK_TOOL}
      ${_BITCODE_FILES}
      "-o"
      "${_OUT}"
    DEPENDS
      ${_LINK_TOOL}
      ${_BITCODE_FILES}
    COMMENT
      "Linking bitcode to ${_OUT}"
    VERBATIM
  )

  # Only add iree_${NAME} as custom target doesn't support aliasing to
  # iree::${NAME}.
  iree_package_name(_PACKAGE_NAME)
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_OUT}"
  )
endfunction()

# iree_link_bitcode()
#
# Builds an LLVM bitcode library from an input file via clang
#
# Parameters:
# NAME: Name of target (see Note).
# SRCS: Source files to pass to clang.
# OUT: Output file name (defaults to NAME.bc).
function(iree_link_bitcode)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT"
    "SRCS"
    ${ARGN}
  )

  set(_LINK_TOOL "$<TARGET_FILE:${IREE_LLVM_LINK_TARGET}>")

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}.bc")
  endif()

  set(_BITCODE_FILES "${_RULE_SRCS}")

  add_custom_command(
    OUTPUT
      ${_OUT}
    COMMAND
      ${_LINK_TOOL}
      ${_BITCODE_FILES}
      "-o"
      "${_OUT}"
    DEPENDS
      ${_LINK_TOOL}
      ${_BITCODE_FILES}
    COMMENT
      "Linking bitcode to ${_OUT}"
    VERBATIM
  )

  # Only add iree_${NAME} as custom target doesn't support aliasing to
  # iree::${NAME}.
  iree_package_name(_PACKAGE_NAME)
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_OUT}"
  )
endfunction()
