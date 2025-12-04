# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# iree_tablegen_library()
#
# Runs iree-tablegen to produce some artifacts.
#
# Parameters:
#   NAME: Name of the tablegen library target.
#   TBLGEN: Tablegen executable to use (IREE or MLIR).
#   TD_FILE: Main .td file to process.
#   OUTS: List of output files and their generation flags.
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

  # Get tablegen executable and target based on TBLGEN parameter.
  if(${_RULE_TBLGEN} MATCHES "IREE")
    set(_TBLGEN_EXE ${IREE_TABLEGEN_EXE})
    set(_TBLGEN_TARGET ${IREE_TABLEGEN_TARGET})
  else()
    set(_TBLGEN_EXE ${MLIR_TABLEGEN_EXE})
    set(_TBLGEN_TARGET ${MLIR_TABLEGEN_TARGET})
  endif()

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

  # Build absolute path for the main .td file.
  if(IS_ABSOLUTE ${_RULE_TD_FILE})
    set(_TD_FILE_ABS ${_RULE_TD_FILE})
  else()
    set(_TD_FILE_ABS ${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_TD_FILE})
  endif()

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

    set(_OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/${_FILE})

    # Create add_custom_command with depfile support for Ninja.
    if(CMAKE_GENERATOR MATCHES "Ninja")
      add_custom_command(
        OUTPUT ${_OUTPUT_FILE}
        COMMAND ${_TBLGEN_EXE} ${_COMMAND} ${_INCLUDE_DIRS} ${_TD_FILE_ABS}
                --write-if-changed -o ${_OUTPUT_FILE} -d ${_OUTPUT_FILE}.d
        DEPENDS ${_TBLGEN_TARGET} ${_TBLGEN_EXE} ${_TD_FILE_ABS}
        DEPFILE ${_OUTPUT_FILE}.d
        COMMENT "Building ${_FILE}..."
        )
    else()
      add_custom_command(
        OUTPUT ${_OUTPUT_FILE}
        COMMAND ${_TBLGEN_EXE} ${_COMMAND} ${_INCLUDE_DIRS} ${_TD_FILE_ABS}
                --write-if-changed -o ${_OUTPUT_FILE}
        DEPENDS ${_TBLGEN_TARGET} ${_TBLGEN_EXE} ${_TD_FILE_ABS}
        COMMENT "Building ${_FILE}..."
        )
    endif()

    list(APPEND _OUTPUTS ${_OUTPUT_FILE})
    set_source_files_properties(${_OUTPUT_FILE} PROPERTIES GENERATED 1)
  endwhile()
  add_custom_target(${_NAME}_target DEPENDS ${_OUTPUTS})
  set_target_properties(${_NAME}_target PROPERTIES FOLDER "Tablegenning")

  add_library(${_NAME} INTERFACE)
  add_dependencies(${_NAME} ${_NAME}_target)

  # Alias the iree_package_name library to iree::package::name.
  iree_package_ns(_PACKAGE_NS)
  iree_add_alias_library(${_PACKAGE_NS}::${_RULE_NAME} ${_NAME})
endfunction()
