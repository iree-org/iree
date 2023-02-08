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
# SRCS: Source files to pass to clang.
# HDRS: Additional headers included by the source files.
# COPTS: additional flags to pass to clang.
# DEFINES: Preprocessor definitions to pass to clang.
# DATA: Additional data required during compilation.
# OUT: Output file name (defaults to NAME.bc).
# PUBLIC: Add this so that this library will be exported under ${PACKAGE}::
#     Also in IDE, target will appear in ${PACKAGE} folder while non PUBLIC
#     will be in ${PACKAGE}/internal.
# TESTONLY: When added, this target will only be built if IREE_BUILD_TESTS=ON.
function(iree_bitcode_library)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY"
    "NAME;OUT"
    "SRCS;HDRS;COPTS;DEFINES;DATA"
    ${ARGN}
  )

  set(_CLANG_TOOL "$<TARGET_FILE:${IREE_CLANG_TARGET}>")
  set(_LINK_TOOL "$<TARGET_FILE:${IREE_LLVM_LINK_TARGET}>")

  # These are copied as part of the clang build; we could allow the user to
  # override this but it should be harmless.
  set(_BUILTIN_HEADERS_PATH "${IREE_BINARY_DIR}/third_party/llvm-project/llvm/lib/clang/${CLANG_VERSION_MAJOR}/include/")

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  if(DEFINED _RULE_OUT)
    set(_OUT "${_RULE_OUT}")
  else()
    set(_OUT "${_RULE_NAME}.bc")
  endif()

  set(_ARGS "-isystem ${_BUILTIN_HEADERS_PATH}")
  list(APPEND _ARGS "${_RULE_COPTS}")
  foreach(_DEFINE ${_RULE_DEFINES})
    list(APPEND _ARGS "-D${_DEFINE}")
  endforeach()

  set(_BITCODE_FILES)
  foreach(_BITCODE_SRC ${_RULE_SRCS})
    get_filename_component(_BITCODE_SRC_PATH "${_BITCODE_SRC}" REALPATH)
    set(_BITCODE_FILE "${_RULE_NAME}_${_BITCODE_SRC}.bc")
    list(APPEND _BITCODE_FILES ${_BITCODE_FILE})
    add_custom_command(
      OUTPUT
        ${_BITCODE_FILE}
      COMMAND
        ${_CLANG_TOOL}
        ${_ARGS}
        "${_BITCODE_SRC_PATH}"
        "-o"
        "${_BITCODE_FILE}"
      DEPENDS
        ${_CLANG_TOOL}
        ${_BITCODE_SRC}
      COMMENT
        "Compiling ${_BITCODE_SRC} to ${_BITCODE_FILE}"
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
      ${_RULE_SRCS}
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
