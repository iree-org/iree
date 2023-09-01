# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_tablegen_library()
#
# Runs iree-tablegen to produce some artifacts.
function(iree_tablegen_library)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "NAME;TBLGEN"
    "TD_FILE;OUTS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Prefix the library with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  if(${_RULE_TBLGEN} MATCHES "IREE")
    set(_TBLGEN "IREE")
  else()
    set(_TBLGEN "MLIR")
  endif()

  set(LLVM_TARGET_DEFINITIONS ${_RULE_TD_FILE})
  set(_INCLUDE_DIRS
    "${MLIR_INCLUDE_DIRS}"
    "${IREE_SOURCE_DIR}/compiler/src"
    "${IREE_BINARY_DIR}/compiler/src"
  )
  if(DEFINED IREE_COMPILER_TABLEGEN_INCLUDE_DIRS)
    list(APPEND _INCLUDE_DIRS ${IREE_COMPILER_TABLEGEN_INCLUDE_DIRS})
  endif()
  list(APPEND _INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
  list(TRANSFORM _INCLUDE_DIRS PREPEND "-I")
  set(_OUTPUTS)
  while(_RULE_OUTS)
    # Eat any number of flags (--a --b ...) and a single file.
    # Flags only impact the successively declared file.
    set(_COMMAND)
    set(_FILE)
    while(_RULE_OUTS AND NOT _FILE)
      list(GET _RULE_OUTS 0 _PART)
      list(REMOVE_AT _RULE_OUTS 0)
      if(${_PART} MATCHES "^-.*")
        # Flag (- or --).
        list(APPEND _COMMAND ${_PART})
      else()
        # File path.
        set(_FILE ${_PART})
      endif()
    endwhile()
    tablegen(${_TBLGEN} ${_FILE} ${_COMMAND} ${_INCLUDE_DIRS})
    list(APPEND _OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/${_FILE})
  endwhile()
  add_custom_target(${_NAME}_target DEPENDS ${_OUTPUTS})
  set_target_properties(${_NAME}_target PROPERTIES FOLDER "Tablegenning")

  add_library(${_NAME} INTERFACE)
  add_dependencies(${_NAME} ${_NAME}_target)

  # Alias the iree_package_name library to iree::package::name.
  iree_package_ns(_PACKAGE_NS)
  add_library(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})
endfunction()
