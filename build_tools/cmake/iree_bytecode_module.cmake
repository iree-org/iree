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
# FLAGS: Flags to pass to the compiler tool (list of strings).
# COMPILE_TOOL: Compiler tool to invoke (CMake target). The default tool is
#     "iree-compile".
# C_IDENTIFIER: Identifier to use for generate c embed code.
#     If omitted then no C embed code will be generated.
# PUBLIC: Add this so that this library will be exported under ${PACKAGE}::
#     Also in IDE, target will appear in ${PACKAGE} folder while non PUBLIC
#     will be in ${PACKAGE}/internal.
# DEPS: Library dependencies to add to the generated embed cc library.
# TESTONLY: When added, this target will only be built if IREE_BUILD_TESTS=ON.
# MODULE_FILE_NAME: Optional. When specified, sets the output bytecode module
#    file name. When not specified, a default file name will be generated from
#    ${NAME}.
# DEPENDS: Optional. Additional dependencies beyond SRC and the tools.
# FRIENDLY_NAME: Optional. Name to use to display build progress info.
#
# Note:
# By default, iree_bytecode_module will create a module target named ${NAME} and
# a library named ${NAME}_c. The library has an alias target iree::${NAME}_c.
# The module doesn't have an alias due to a lack of support on custom target.
# The iree:: form should always be used to reduce namespace pollution.
function(iree_bytecode_module)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY"
    "NAME;SRC;COMPILE_TOOL;C_IDENTIFIER;MODULE_FILE_NAME;FRIENDLY_NAME"
    "FLAGS;DEPENDS;DEPS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Set default for COMPILE_TOOL.
  if(DEFINED _RULE_COMPILE_TOOL)
    set(_COMPILE_TOOL ${_RULE_COMPILE_TOOL})
  else()
    set(_COMPILE_TOOL "iree-compile")
  endif()

  if(DEFINED _RULE_MODULE_FILE_NAME)
    set(_MODULE_FILE_NAME "${_RULE_MODULE_FILE_NAME}")
  else()
    set(_MODULE_FILE_NAME "${_RULE_NAME}.vmfb")
  endif()

  iree_get_executable_path(_COMPILE_TOOL_EXECUTABLE ${_COMPILE_TOOL})

  set(_ARGS "${_RULE_FLAGS}")

  get_filename_component(_SRC_PATH "${_RULE_SRC}" REALPATH)
  list(APPEND _ARGS "${_SRC_PATH}")
  list(APPEND _ARGS "-o")
  list(APPEND _ARGS "${_MODULE_FILE_NAME}")

  # If an LLVM CPU backend is enabled, supply the linker tool.
  if(IREE_LLD_TARGET)
    iree_get_executable_path(_LINKER_TOOL_EXECUTABLE "lld")
    list(APPEND _ARGS "--iree-llvm-embedded-linker-path=\"${_LINKER_TOOL_EXECUTABLE}\"")
    list(APPEND _ARGS "--iree-llvm-wasm-linker-path=\"${_LINKER_TOOL_EXECUTABLE}\"")
    # Note: --iree-llvm-system-linker-path is left unspecified.
  endif()

  if(IREE_BYTECODE_MODULE_FORCE_SYSTEM_DYLIB_LINKER)
    list(APPEND _ARGS "--iree-llvm-link-embedded=false")
  endif()

  # Support testing in TSan build dirs. Unlike other sanitizers, TSan is an
  # ABI break: when the host code is built with TSan, the module must be too,
  # otherwise we get crashes calling module code.
  if(IREE_BYTECODE_MODULE_ENABLE_TSAN)
    list(APPEND _ARGS "--iree-llvm-sanitize=thread")
  endif()

  if(_RULE_FRIENDLY_NAME)
    set(_FRIENDLY_NAME "${_RULE_FRIENDLY_NAME}")
  else()
    get_filename_component(_FRIENDLY_NAME "${_RULE_SRC}" NAME)
  endif()

  # Depending on the binary instead of the target here given we might not have
  # a target in this CMake invocation when cross-compiling.
  add_custom_command(
    OUTPUT
      "${_MODULE_FILE_NAME}"
    COMMAND
      ${_COMPILE_TOOL_EXECUTABLE}
      ${_ARGS}
    # Changes to either the compiler tool or the input sources should rebuild.
    DEPENDS
      ${_COMPILE_TOOL_EXECUTABLE}
      ${_LINKER_TOOL_EXECUTABLE}
      ${_RULE_SRC}
      ${_RULE_DEPENDS}
    COMMENT
      "Generating ${_MODULE_FILE_NAME} from ${_FRIENDLY_NAME}"
    VERBATIM
  )

  # Only add iree_${NAME} as custom target doesn't support aliasing to
  # iree::${NAME}.
  iree_package_name(_PACKAGE_NAME)
  add_custom_target("${_PACKAGE_NAME}_${_RULE_NAME}"
    DEPENDS "${_MODULE_FILE_NAME}"
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
      DEPS
        ${_RULE_DEPS}
    )
  endif()
endfunction()
