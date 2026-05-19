# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_vmasm_module()
#
# Builds an IREE bytecode module from VM assembly.
#
# Parameters:
# NAME: Name of target.
# SRC: Source file to assemble into a bytecode module.
# MODULE_FILE_NAME: Optional output bytecode module file name. Defaults to
#     ${NAME}.vmfb.
# ASSEMBLE_TOOL: Assembler tool target or executable path to invoke. Defaults
#     to iree-as-module.
# C_IDENTIFIER: Identifier to use for generated C embed code. If omitted then
#     no C embed code will be generated.
# DEPS: Library dependencies to add to the generated embed cc library.
# TESTONLY: When added, this target will only be built if IREE_BUILD_TESTS=ON.
# PUBLIC: Add this so that generated libraries are exported under ${PACKAGE}::.
function(iree_vmasm_module)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY"
    "NAME;SRC;MODULE_FILE_NAME;ASSEMBLE_TOOL;C_IDENTIFIER"
    "DEPS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  if(DEFINED _RULE_ASSEMBLE_TOOL)
    set(_ASSEMBLE_TOOL ${_RULE_ASSEMBLE_TOOL})
  elseif(IREE_HOST_BIN_DIR)
    set(_ASSEMBLE_TOOL "${IREE_HOST_BIN_DIR}/iree-as-module")
  else()
    set(_ASSEMBLE_TOOL "iree-as-module")
  endif()

  if(TARGET "${_ASSEMBLE_TOOL}")
    set(_ASSEMBLE_TOOL_COMMAND "$<TARGET_FILE:${_ASSEMBLE_TOOL}>")
  else()
    set(_ASSEMBLE_TOOL_COMMAND "${_ASSEMBLE_TOOL}")
  endif()

  if(DEFINED _RULE_MODULE_FILE_NAME)
    set(_MODULE_FILE_NAME "${_RULE_MODULE_FILE_NAME}")
  else()
    set(_MODULE_FILE_NAME "${_RULE_NAME}.vmfb")
  endif()

  get_filename_component(_SRC_PATH "${_RULE_SRC}" REALPATH)
  add_custom_command(
    OUTPUT
      "${_MODULE_FILE_NAME}"
    COMMAND
      ${_ASSEMBLE_TOOL_COMMAND}
      "--output=${_MODULE_FILE_NAME}"
      "${_SRC_PATH}"
    DEPENDS
      "${_SRC_PATH}"
      ${_ASSEMBLE_TOOL}
    COMMENT
      "Assembling IREE VM module ${_RULE_NAME}"
    VERBATIM
  )

  iree_package_name(_PACKAGE_NAME)
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS
      "${_MODULE_FILE_NAME}"
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
      SRCS
        "${_MODULE_FILE_NAME}"
      C_FILE_OUTPUT
        "${_RULE_NAME}_c.c"
      H_FILE_OUTPUT
        "${_RULE_NAME}_c.h"
      IDENTIFIER
        "${_RULE_C_IDENTIFIER}"
      FLATTEN
        "${_PUBLIC_ARG}"
        "${_TESTONLY_ARG}"
      DEPS
        ${_RULE_DEPS}
    )
  endif()
endfunction()
