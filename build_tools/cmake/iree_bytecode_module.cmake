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
# By default, iree_bytecode_module will create a library named ${NAME}_c,
# and alias target iree::${NAME}_c. The iree:: form should always be used.
# This is to reduce namespace pollution.
function(iree_bytecode_module)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY"
    "NAME;SRC;TRANSLATE_TOOL;C_IDENTIFIER;OPT_TOOL;MODULE_FILE_NAME"
    "FLAGS;OPT_FLAGS"
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

  if(DEFINED _RULE_MODULE_FILE_NAME)
    set(_MODULE_FILE_NAME "${_RULE_MODULE_FILE_NAME}")
  else()
    set(_MODULE_FILE_NAME "${_RULE_NAME}.vmfb")
  endif()

  # If OPT_FLAGS was specified, preprocess the source file with the OPT_TOOL
  if(_RULE_OPT_FLAGS)
    # Create the filename for the output of OPT_TOOL, which
    # will relace _RULE_SRC as the input to iree_bytecode_module.
    set(_TRANSLATE_SRC_BASENAME "${_RULE_NAME}.opt.mlir")
    set(_TRANSLATE_SRC "${CMAKE_CURRENT_BINARY_DIR}/${_TRANSLATE_SRC_BASENAME}")

    # Set default for OPT_TOOL.
    if(_RULE_OPT_TOOL)
      set(_OPT_TOOL ${_RULE_OPT_TOOL})
    else()
      set(_OPT_TOOL "iree-opt")
    endif()

    # Prepare the OPT_TOOL command line.
    iree_get_executable_path(_OPT_TOOL_EXECUTABLE ${_OPT_TOOL})

    set(_ARGS "${_RULE_OPT_FLAGS}")
    get_filename_component(_SRC_PATH "${_RULE_SRC}" REALPATH)
    list(APPEND _ARGS "${_SRC_PATH}")
    list(APPEND _ARGS "-o")
    list(APPEND _ARGS "${_TRANSLATE_SRC}")

    add_custom_command(
      OUTPUT
        "${_TRANSLATE_SRC_BASENAME}"
      COMMAND
        ${_OPT_TOOL_EXECUTABLE}
        ${_ARGS}
      # Changes to the opt tool should trigger rebuilding.
      # Using {_OPT_TOOL} as the dependency would only work when the tools
      # are built in the same cmake build directory as the tests, that is,
      # when NOT cross-compiling. Using {_OPT_TOOL_EXECUTABLE} works
      # uniformly regardless of that.
      DEPENDS
        ${_OPT_TOOL_EXECUTABLE}
        ${_RULE_SRC}
      VERBATIM
    )
  else()
    # OPT_FLAGS was not specified, so are not using the OPT_TOOL.
    # Just pass the source file directly as the source for the bytecode module.
    set(_TRANSLATE_SRC "${_RULE_SRC}")
  endif()

  iree_get_executable_path(_TRANSLATE_TOOL_EXECUTABLE ${_TRANSLATE_TOOL})
  iree_get_executable_path(_EMBEDDED_LINKER_TOOL_EXECUTABLE "lld")

  set(_ARGS "${_RULE_FLAGS}")

  get_filename_component(_TRANSLATE_SRC_PATH "${_TRANSLATE_SRC}" REALPATH)
  list(APPEND _ARGS "${_TRANSLATE_SRC_PATH}")
  list(APPEND _ARGS "-o")
  list(APPEND _ARGS "${_MODULE_FILE_NAME}")
  list(APPEND _ARGS "-iree-llvm-embedded-linker-path=\"${_EMBEDDED_LINKER_TOOL_EXECUTABLE}\"")

  # Depending on the binary instead of the target here given we might not have
  # a target in this CMake invocation when cross-compiling.
  add_custom_command(
    OUTPUT
      "${_MODULE_FILE_NAME}"
    COMMAND
      ${_TRANSLATE_TOOL_EXECUTABLE}
      ${_ARGS}
    # Changes to either the translation tool or the input source should
    # trigger rebuilding.
    DEPENDS
      ${_TRANSLATE_TOOL_EXECUTABLE}
      ${_EMBEDDED_LINKER_TOOL_EXECUTABLE}
      ${_TRANSLATE_SRC}
    VERBATIM
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
