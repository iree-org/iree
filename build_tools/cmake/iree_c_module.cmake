# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_c_module()
#
# Parameters:
# NAME: Name of target (see Note).
# SRC: MLIR source file to compile into a c module.
# H_FILE_OUTPUT: The H header file to output.
# COMPILE_TOOL: Compiler tool to invoke (CMake target). The default tool is
#     "iree-compile".
# FLAGS: Additional flags to pass to the compile tool (list of strings).
#     `--output-format=vm-c` is included automatically.
# TESTONLY: When added, this target will only be built if user passes
#     -DIREE_BUILD_TESTS=ON to CMake.
#
# Note:
# By default, iree_c_module will create a library named ${NAME},
# and alias target iree::${NAME}. The iree:: form should always be used.
# This is to reduce namespace pollution.
function(iree_c_module)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "NAME;SRC;H_FILE_OUTPUT;COMPILE_TOOL"
    "FLAGS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Replace dependencies passed by ::name with iree::package::name
  iree_package_ns(_PACKAGE_NS)
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  # Prefix the library with the package name, so we get: iree_package_name.
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}_hdrs")

  # Set defaults for COMPILE_TOOL.
  if(DEFINED _RULE_COMPILE_TOOL)
    set(_COMPILE_TOOL ${_RULE_COMPILE_TOOL})
  else()
    set(_COMPILE_TOOL "iree-compile")
  endif()

  iree_get_executable_path(_COMPILE_TOOL_EXECUTABLE ${_COMPILE_TOOL})

  set(_ARGS "--output-format=vm-c")
  list(APPEND _ARGS "${_RULE_FLAGS}")
  list(APPEND _ARGS "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRC}")
  list(APPEND _ARGS "-o")
  list(APPEND _ARGS "${_RULE_H_FILE_OUTPUT}")

  add_custom_command(
    OUTPUT "${_RULE_H_FILE_OUTPUT}"
    COMMAND ${_COMPILE_TOOL_EXECUTABLE} ${_ARGS}
    # Changes to either the compiler tool or the input source should rebuild.
    DEPENDS ${_COMPILE_TOOL_EXECUTABLE} ${_RULE_SRC}
  )

  iree_cc_library(
    NAME ${_RULE_NAME}
    HDRS "${_RULE_H_FILE_OUTPUT}"
    SRCS "${IREE_SOURCE_DIR}/runtime/src/iree/vm/module_impl_emitc.c"
    INCLUDES "${CMAKE_CURRENT_BINARY_DIR}"
    COPTS
      "-DEMITC_IMPLEMENTATION=\"${_RULE_H_FILE_OUTPUT}\""
      "${_TESTONLY_ARG}"
    DEPS
      # Include paths and options for the runtime sources.
      iree::runtime::src::defs
  )

  set(_GEN_TARGET "${_NAME}_gen")
  add_custom_target(
    ${_GEN_TARGET}
    DEPENDS
      ${_RULE_H_FILE_OUTPUT}
  )

  add_library(${_NAME} INTERFACE)
  add_dependencies(${_NAME} ${_GEN_TARGET})
  add_dependencies(${_NAME}
    iree::vm
    iree::vm::ops
    iree::vm::ops_emitc
    iree::vm::shims_emitc
  )

  # Alias the iree_package_name library to iree::package::name.
  # This lets us more clearly map to Bazel and makes it possible to
  # disambiguate the underscores in paths vs. the separators.
  add_library(${_PACKAGE_NS}::${_RULE_NAME}_hdrs ALIAS ${_NAME})
  iree_package_dir(_PACKAGE_DIR)
  if(${_RULE_NAME}_hdrs STREQUAL ${_PACKAGE_DIR})
    # If the library name matches the package then treat it as a default.
    # For example, foo/bar/ library 'bar' would end up as 'foo::bar'.
    add_library(${_PACKAGE_NS} ALIAS ${_NAME})
  endif()
endfunction()
