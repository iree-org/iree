# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_bytecode_module()
#
# CMake function to imitate Bazel's iree_bytecode_module rule.
#
# Parameters:
# NAME: Name of target (see Note).
# SRC: Source file to compile into a bytecode module.
# FLAGS: Flags to pass to the translation tool (list of strings).
# TRANSLATE_TOOL: Translation tool to invoke (CMake target). The default
#     tool is "iree-translate".
# C_IDENTIFIER: Identifier to use for generate c embed code.
#     If omitted then no C embed code will be generated.
# PUBLIC: Add this so that this library will be exported under ${PACKAGE}::
#     Also in IDE, target will appear in ${PACKAGE} folder while non PUBLIC
#     will be in ${PACKAGE}/internal.
# TESTONLY: When added, this target will only be built if user passes
#    -DIREE_BUILD_TESTS=ON to CMake.
#
# Note:
# By default, iree_bytecode_module will create a library named ${NAME}_cc,
# and alias target iree::${NAME}_cc. The iree:: form should always be used.
# This is to reduce namespace pollution.
function(iree_bytecode_module)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY"
    "NAME;SRC;TRANSLATE_TOOL;C_IDENTIFIER"
    "FLAGS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Set default for TRANSLATE_TOOL.
  if(DEFINED _RULE_TRANSLATE_TOOL)
    set(_TRANSLATE_TOOL ${_RULE_TRANSLATE_TOOL})
  else()
    set(_TRANSLATE_TOOL "iree-translate")
  endif()

  iree_get_executable_path(_TRANSLATE_TOOL_EXECUTABLE ${_TRANSLATE_TOOL})
  iree_get_executable_path(_EMBEDDED_LINKER_TOOL_EXECUTABLE "lld")

  set(_ARGS "${_RULE_FLAGS}")
  # Check source file path to support absolute path.
  set(_SRC_PATH "${_RULE_SRC}")
  if (NOT IS_ABSOLUTE ${_SRC_PATH})
    set(_SRC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRC}")
  endif()
  list(APPEND _ARGS "${_SRC_PATH}")
  list(APPEND _ARGS "-o")
  list(APPEND _ARGS "${_RULE_NAME}.vmfb")
  list(APPEND _ARGS "-iree-llvm-embedded-linker-path=${_EMBEDDED_LINKER_TOOL_EXECUTABLE}")

  # Depending on the binary instead of the target here given we might not have
  # a target in this CMake invocation when cross-compiling.
  add_custom_command(
    OUTPUT
      "${_RULE_NAME}.vmfb"
    COMMAND
      ${_TRANSLATE_TOOL_EXECUTABLE}
      ${_ARGS}
    # Changes to either the translation tool or the input source should
    # trigger rebuilding.
    DEPENDS
      ${_TRANSLATE_TOOL_EXECUTABLE}
      ${_EMBEDDED_LINKER_TOOL_EXECUTABLE}
      ${_RULE_SRC}
  )

  if(_RULE_TESTONLY)
    set(_TESTONLY_ARG "TESTONLY")
  endif()
  if(_RULE_PUBLIC)
    set(_PUBLIC_ARG "PUBLIC")
  endif()

  if(_RULE_C_IDENTIFIER)
    iree_c_embed_data(
      NAME
        "${_RULE_NAME}_c"
      IDENTIFIER
        "${_RULE_C_IDENTIFIER}"
      GENERATED_SRCS
        "${_RULE_NAME}.vmfb"
      C_FILE_OUTPUT
        "${_RULE_NAME}_c.c"
      H_FILE_OUTPUT
        "${_RULE_NAME}_c.h"
      FLATTEN
        "${_PUBLIC_ARG}"
        "${_TESTONLY_ARG}"
    )
  endif()
endfunction()
