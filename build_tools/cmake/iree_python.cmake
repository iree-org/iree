# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

###############################################################################
# Main user rules
###############################################################################

# iree_py_library()
#
# CMake function to imitate Bazel's iree_py_library rule.
#
# Parameters:
# NAME: name of target
# SRCS: List of source files for the library
# DEPS: List of other targets the test python libraries require
# PYEXT_DEPS: List of deps of extensions built with iree_pyext_module
function(iree_py_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DEPS;PYEXT_DEPS"
    ${ARGN}
  )

  iree_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::iree::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  add_custom_target(${_NAME} ALL
    DEPENDS ${_RULE_DEPS}
  )

  # Symlink each file as its own target.
  foreach(_SRC_FILE ${_RULE_SRCS})
    # _SRC_FILE could have other path components in it, so we need to make a
    # directory for it. Ninja does this automatically, but make doesn't. See
    # https://github.com/openxla/iree/issues/6801
    set(_SRC_BIN_PATH "${CMAKE_CURRENT_BINARY_DIR}/${_SRC_FILE}")
    get_filename_component(_SRC_BIN_DIR "${_SRC_BIN_PATH}" DIRECTORY)
    add_custom_command(
      TARGET ${_NAME}
      COMMAND
        ${CMAKE_COMMAND} -E make_directory "${_SRC_BIN_DIR}"
      COMMAND ${CMAKE_COMMAND} -E create_symlink
        "${CMAKE_CURRENT_SOURCE_DIR}/${_SRC_FILE}" "${_SRC_BIN_PATH}"
      BYPRODUCTS "${_SRC_BIN_PATH}"
    )
  endforeach()

  # Add PYEXT_DEPS if any.
  if(_RULE_PYEXT_DEPS)
    list(TRANSFORM _RULE_PYEXT_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
    add_dependencies(${_NAME} ${_RULE_PYEXT_DEPS})
  endif()
endfunction()

# iree_local_py_test()
#
# CMake function to run python test with provided python package paths.
#
# Parameters:
# NAME: name of test
# SRC: Test source file
# ARGS: Command line arguments to the Python source file.
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
# GENERATED_IN_BINARY_DIR: If present, indicates that the srcs have been
#   in the CMAKE_CURRENT_BINARY_DIR.
# PACKAGE_DIRS: Python package paths to be added to PYTHONPATH.
function(iree_local_py_test)
  if(NOT IREE_BUILD_TESTS OR ANDROID OR EMSCRIPTEN)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    "GENERATED_IN_BINARY_DIR"
    "NAME;SRC"
    "ARGS;LABELS;PACKAGE_DIRS;TIMEOUT"
    ${ARGN}
  )

  # Switch between source and generated tests.
  set(_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  if(_RULE_GENERATED_IN_BINARY_DIR)
    set(_SRC_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_NAME_PATH "${_PACKAGE_PATH}/${_RULE_NAME}")
  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")

  add_test(
    NAME ${_NAME_PATH}
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRC}"
      ${_RULE_ARGS}
  )

  set_property(TEST ${_NAME_PATH} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST ${_NAME_PATH} PROPERTY TIMEOUT ${_RULE_ARGS})

  # Extend the PYTHONPATH environment variable with _RULE_PACKAGE_DIRS.
  list(APPEND _RULE_PACKAGE_DIRS "$ENV{PYTHONPATH}")
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    # Windows uses semi-colon delimiters, but so does CMake, so escape them.
    list(JOIN _RULE_PACKAGE_DIRS "\\;" _PYTHONPATH)
  else()
    list(JOIN _RULE_PACKAGE_DIRS ":" _PYTHONPATH)
  endif()
  set_property(TEST ${_NAME_PATH} PROPERTY ENVIRONMENT
      "PYTHONPATH=${_PYTHONPATH}"
  )

  if (NOT DEFINED _RULE_TIMEOUT)
    set(_RULE_TIMEOUT 60)
  endif()

  iree_configure_test(${_NAME_PATH})

  # TODO(marbre): Find out how to add deps to tests.
  #               Similar to _RULE_DATA in iree_lit_test().
endfunction()

# iree_py_test()
#
# CMake function to imitate Bazel's iree_py_test rule.
#
# Parameters:
# NAME: name of test
# SRCS: Test source file (single file only, despite name)
# ARGS: Command line arguments to the Python source file.
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
# GENERATED_IN_BINARY_DIR: If present, indicates that the srcs have been
#   in the CMAKE_CURRENT_BINARY_DIR.
function(iree_py_test)
  cmake_parse_arguments(
    _RULE
    "GENERATED_IN_BINARY_DIR"
    "NAME;SRCS"
    "ARGS;LABELS;TIMEOUT"
    ${ARGN}
  )
  if(NOT IREE_BUILD_PYTHON_BINDINGS)
    return()
  endif()

  iree_local_py_test(
    NAME
      "${_RULE_NAME}"
    SRC
      "${_RULE_SRCS}"
    ARGS
      ${_RULE_ARGS}
    LABELS
      ${_RULE_LABELS}
    PACKAGE_DIRS
      "${IREE_BINARY_DIR}/compiler/bindings/python"
      "${IREE_BINARY_DIR}/runtime/bindings/python"
    GENERATED_IN_BINARY_DIR
      "${_RULE_GENERATED_IN_BINARY_DIR}"
    TIMEOUT
      ${_RULE_TIMEOUT}
  )
endfunction()

# iree_build_tools_py_test()
#
# CMake function to test with build_tools python modules.
#
# Parameters:
# NAME: name of test
# SRC: Test source file
# ARGS: Command line arguments to the Python source file.
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
# PACKAGE_DIRS: Additional python module paths.
function(iree_build_tools_py_test)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC"
    "ARGS;LABELS;PACKAGE_DIRS"
    ${ARGN}
  )

  iree_local_py_test(
    NAME
      "${_RULE_NAME}"
    SRC
      "${_RULE_SRC}"
    ARGS
      ${_RULE_ARGS}
    LABELS
      ${_RULE_LABELS}
    PACKAGE_DIRS
      ${_RULE_PACKAGE_DIRS}
      "${IREE_ROOT_DIR}/build_tools/python"
  )
endfunction()
