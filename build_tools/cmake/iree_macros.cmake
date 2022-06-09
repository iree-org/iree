# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

#-------------------------------------------------------------------------------
# Missing CMake Variables
#-------------------------------------------------------------------------------

if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Windows")
  set(IREE_HOST_SCRIPT_EXT "bat")
  # https://gitlab.kitware.com/cmake/cmake/-/issues/17553
  set(IREE_HOST_EXECUTABLE_SUFFIX ".exe")
else()
  set(IREE_HOST_SCRIPT_EXT "sh")
  set(IREE_HOST_EXECUTABLE_SUFFIX "")
endif()

#-------------------------------------------------------------------------------
# General utilities
#-------------------------------------------------------------------------------

# iree_to_bool
#
# Sets `variable` to `ON` if `value` is true and `OFF` otherwise.
function(iree_to_bool VARIABLE VALUE)
  if(VALUE)
    set(${VARIABLE} "ON" PARENT_SCOPE)
  else()
    set(${VARIABLE} "OFF" PARENT_SCOPE)
  endif()
endfunction()

# iree_append_list_to_string
#
# Joins ${ARGN} together as a string separated by " " and appends it to
# ${VARIABLE}.
function(iree_append_list_to_string VARIABLE)
  if(NOT "${ARGN}" STREQUAL "")
    string(JOIN " " _ARGN_STR ${ARGN})
    set(${VARIABLE} "${${VARIABLE}} ${_ARGN_STR}" PARENT_SCOPE)
  endif()
endfunction()


#-------------------------------------------------------------------------------
# Packages and Paths
#-------------------------------------------------------------------------------

# Sets ${PACKAGE_NS} to the IREE-root relative package name in C++ namespace
# format (::).
#
# Examples:
#   compiler/src/iree/compiler/Utils/CMakeLists.txt -> iree::compiler::Utils
#   runtime/src/iree/base/CMakeLists.txt -> iree::base
#   tests/e2e/CMakeLists.txt -> iree::tests::e2e
function(iree_package_ns PACKAGE_NS)
  # Get the relative path of the current dir (i.e. runtime/src/iree/vm).
  string(REPLACE ${IREE_ROOT_DIR} "" _RELATIVE_PATH ${CMAKE_CURRENT_LIST_DIR})
  string(SUBSTRING ${_RELATIVE_PATH} 1 -1 _RELATIVE_PATH)

  # If changing the directory/package mapping rules, please also implement
  # the corresponding rule in:
  #   build_tools/bazel_to_cmake/bazel_to_cmake_targets.py
  # Some sub-trees form their own roots for package purposes. Rewrite them.
  if(_RELATIVE_PATH MATCHES "^compiler/src/(.*)")
    # compiler/src/iree/compiler -> iree/compiler
    set(_PACKAGE "${CMAKE_MATCH_1}")
  elseif(_RELATIVE_PATH MATCHES "^runtime/src/(.*)")
    # runtime/src/iree/base -> iree/base
    set(_PACKAGE "${CMAKE_MATCH_1}")
  elseif(_RELATIVE_PATH MATCHES "^tools$")
    # Special case for tools/ -> "" (empty string)
    # For example, tools/iree-compile -> iree-compile (no namespace)
    set(_PACKAGE "")
  else()
    # Default to prefixing with iree/
    set(_PACKAGE "iree/${_RELATIVE_PATH}")
  endif()

  string(REPLACE "/" "::" _PACKAGE_NS "${_PACKAGE}")

  if(_DEBUG_IREE_PACKAGE_NAME)
    message(STATUS "iree_package_ns(): map ${_RELATIVE_PATH} -> ${_PACKAGE_NS}")
  endif()

  set(${PACKAGE_NS} ${_PACKAGE_NS} PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_NAME} to the IREE-root relative package name.
#
# Example when called from iree/base/CMakeLists.txt:
#   iree_base
function(iree_package_name PACKAGE_NAME)
  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "_" _PACKAGE_NAME "${_PACKAGE_NS}")
  set(${PACKAGE_NAME} ${_PACKAGE_NAME} PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_PATH} to the IREE-root relative package path.
#
# Example when called from iree/base/CMakeLists.txt:
#   iree/base
function(iree_package_path PACKAGE_PATH)
  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(${PACKAGE_PATH} ${_PACKAGE_PATH} PARENT_SCOPE)
endfunction()

# Sets ${PACKAGE_DIR} to the directory name of the current package.
#
# Example when called from iree/base/CMakeLists.txt:
#   base
function(iree_package_dir PACKAGE_DIR)
  iree_package_ns(_PACKAGE_NS)
  string(FIND "${_PACKAGE_NS}" "::" _END_OFFSET REVERSE)
  math(EXPR _END_OFFSET "${_END_OFFSET} + 2")
  string(SUBSTRING ${_PACKAGE_NS} ${_END_OFFSET} -1 _PACKAGE_DIR)
  set(${PACKAGE_DIR} ${_PACKAGE_DIR} PARENT_SCOPE)
endfunction()

# iree_get_executable_path
#
# Gets the path to an executable in a cross-compilation-aware way. This
# should be used when accessing binaries that are used as part of the build,
# such as for generating files used for later build steps.
#
# Paramters:
# - OUTPUT_PATH_VAR: variable name for receiving the path to the built target.
# - EXECUTABLE: the executable to get its path. Note that this needs to be the
#     name of the executable target when not cross compiling and the basename of
#     the binary when importing a binary from a host build. Thus this should be
#     the global unqualified name of the binary, not the fully-specified name.
function(iree_get_executable_path OUTPUT_PATH_VAR EXECUTABLE)
  if(NOT DEFINED IREE_HOST_BINARY_ROOT OR TARGET "${EXECUTABLE}")
    # We can either expect the target to be defined as part of this CMake
    # invocation (if not cross compiling) or the target is defined already.
    set(${OUTPUT_PATH_VAR} "$<TARGET_FILE:${EXECUTABLE}>" PARENT_SCOPE)
  else()
    # The target won't be directly defined by this CMake invocation so check
    # for an already built executable at IREE_HOST_BINARY_ROOT. If we find it,
    # add it as an imported target so it gets picked up on later invocations.
    set(_EXECUTABLE_PATH "${IREE_HOST_BINARY_ROOT}/bin/${EXECUTABLE}${IREE_HOST_EXECUTABLE_SUFFIX}")
    if(EXISTS ${_EXECUTABLE_PATH})
      add_executable("${EXECUTABLE}" IMPORTED GLOBAL)
      set_property(TARGET "${EXECUTABLE}" PROPERTY IMPORTED_LOCATION "${_EXECUTABLE_PATH}")
      set(${OUTPUT_PATH_VAR} "$<TARGET_FILE:${EXECUTABLE}>" PARENT_SCOPE)
    else()
      message(FATAL_ERROR "Could not find '${EXECUTABLE}' at '${_EXECUTABLE_PATH}'. "
              "Ensure that IREE_HOST_BINARY_ROOT points to installed binaries.")
    endif()
  endif()
endfunction()

#-------------------------------------------------------------------------------
# select()-like Evaluation
#-------------------------------------------------------------------------------

# Appends ${OPTS} with a list of values based on the current compiler.
#
# Example:
#   iree_select_compiler_opts(COPTS
#     CLANG
#       "-Wno-foo"
#       "-Wno-bar"
#     CLANG_CL
#       "/W3"
#     GCC
#       "-Wsome-old-flag"
#     MSVC
#       "/W3"
#   )
#
# Note that variables are allowed, making it possible to share options between
# different compiler targets.
function(iree_select_compiler_opts OPTS)
  cmake_parse_arguments(
    PARSE_ARGV 1
    _IREE_SELECTS
    ""
    ""
    "ALL;CLANG;CLANG_GTE_10;CLANG_CL;MSVC;GCC;CLANG_OR_GCC;MSVC_OR_CLANG_CL"
  )
  # OPTS is a variable containing the *name* of the variable being populated, so
  # we need to dereference it twice.
  set(_OPTS "${${OPTS}}")
  list(APPEND _OPTS "${_IREE_SELECTS_ALL}")
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    list(APPEND _OPTS "${_IREE_SELECTS_GCC}")
    list(APPEND _OPTS "${_IREE_SELECTS_CLANG_OR_GCC}")
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if(MSVC)
      list(APPEND _OPTS ${_IREE_SELECTS_CLANG_CL})
      list(APPEND _OPTS ${_IREE_SELECTS_MSVC_OR_CLANG_CL})
    else()
      list(APPEND _OPTS ${_IREE_SELECTS_CLANG})
      if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
        list(APPEND _OPTS ${_IREE_SELECTS_CLANG_GTE_10})
      endif()
      list(APPEND _OPTS ${_IREE_SELECTS_CLANG_OR_GCC})
    endif()
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    list(APPEND _OPTS ${_IREE_SELECTS_MSVC})
    list(APPEND _OPTS ${_IREE_SELECTS_MSVC_OR_CLANG_CL})
  else()
    message(ERROR "Unknown compiler: ${CMAKE_CXX_COMPILER}")
    list(APPEND _OPTS "")
  endif()
  set(${OPTS} ${_OPTS} PARENT_SCOPE)
endfunction()

#-------------------------------------------------------------------------------
# Data dependencies
#-------------------------------------------------------------------------------

# Adds 'data' dependencies to a target.
#
# Parameters:
# NAME: name of the target to add data dependencies to
# DATA: List of targets and/or files in the source tree (relative to the
# project root).
function(iree_add_data_dependencies)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "DATA"
    ${ARGN}
  )

  if(NOT _RULE_DATA)
    return()
  endif()

  foreach(_DATA_LABEL ${_RULE_DATA})
    if(TARGET ${_DATA_LABEL})
      add_dependencies(${_RULE_NAME} ${_DATA_LABEL})
    else()
      # Not a target, assume to be a file instead.
      set(_FILE_PATH ${_DATA_LABEL})

      # Create a target which copies the data file into the build directory.
      # If this file is included in multiple rules, only create the target once.
      string(REPLACE "::" "_" _DATA_TARGET ${_DATA_LABEL})
      string(REPLACE "/" "_" _DATA_TARGET ${_DATA_TARGET})
      if(NOT TARGET ${_DATA_TARGET})
        set(_INPUT_PATH "${PROJECT_SOURCE_DIR}/${_FILE_PATH}")
        set(_OUTPUT_PATH "${PROJECT_BINARY_DIR}/${_FILE_PATH}")
        add_custom_target(${_DATA_TARGET}
          COMMAND ${CMAKE_COMMAND} -E copy ${_INPUT_PATH} ${_OUTPUT_PATH}
        )
      endif()

      add_dependencies(${_RULE_NAME} ${_DATA_TARGET})
    endif()
  endforeach()
endfunction()

#-------------------------------------------------------------------------------
# Tool symlinks
#-------------------------------------------------------------------------------

# iree_symlink_tool
#
# Adds a command to TARGET which symlinks a tool from elsewhere
# (FROM_TOOL_TARGET_NAME) to a local file name (TO_EXE_NAME) in the current
# binary directory.
#
# Parameters:
#   TARGET: Local target to which to add the symlink command (i.e. an
#     iree_py_library, etc).
#   FROM_TOOL_TARGET: Target of the tool executable that is the source of the
#     link.
#   TO_EXE_NAME: The executable name to output in the current binary dir.
function(iree_symlink_tool)
  cmake_parse_arguments(
    _RULE
    ""
    "TARGET;FROM_TOOL_TARGET;TO_EXE_NAME"
    ""
    ${ARGN}
  )

  # Transform TARGET
  iree_package_ns(_PACKAGE_NS)
  iree_package_name(_PACKAGE_NAME)
  set(_TARGET "${_PACKAGE_NAME}_${_RULE_TARGET}")
  set(_FROM_TOOL_TARGET ${_RULE_FROM_TOOL_TARGET})
  set(_TO_TOOL_PATH "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_TO_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX}")
  get_filename_component(_TO_TOOL_DIR "${_TO_TOOL_PATH}" DIRECTORY)


  add_custom_command(
    TARGET "${_TARGET}"
    BYPRODUCTS
      "${CMAKE_CURRENT_BINARY_DIR}/${_RULE_TO_EXE_NAME}${CMAKE_EXECUTABLE_SUFFIX}"
    COMMAND
      ${CMAKE_COMMAND} -E make_directory "${_TO_TOOL_DIR}"
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink
        "$<TARGET_FILE:${_FROM_TOOL_TARGET}>"
        "${_TO_TOOL_PATH}"
  )
endfunction()


#-------------------------------------------------------------------------------
# Tests
#-------------------------------------------------------------------------------

# iree_add_test_environment_properties
#
# Adds test environment variable properties based on the current build options.
#
# Parameters:
#   TEST_NAME: the test name, e.g. iree/base:math_test
function(iree_add_test_environment_properties TEST_NAME)
  # IREE_*_DISABLE environment variables may used to skip test cases which
  # require both a compiler target backend and compatible runtime HAL driver.
  #
  # These variables may be set by the test environment, typically as a property
  # of some continuous execution test runner or by an individual developer, or
  # here by the build system.
  #
  # Tests which only depend on a compiler target backend or a runtime HAL
  # driver, but not both, should generally use a different method of filtering.
  if(NOT IREE_TARGET_BACKEND_VULKAN_SPIRV OR NOT IREE_HAL_DRIVER_VULKAN)
    set_property(TEST ${TEST_NAME} APPEND PROPERTY ENVIRONMENT "IREE_VULKAN_DISABLE=1")
  endif()
endfunction()

# iree_check_defined
#
# A lightweight way to check that all the given variables are defined. Useful
# in cases like checking that a function has been passed all required arguments.
# Doesn't give usage-specific error messages, but still significantly better
# than no error checking.
# Variable names should be passed directly without quoting or dereferencing.
# Example:
#   iree_check_defined(_SOME_VAR _AND_ANOTHER_VAR)
macro(iree_check_defined)
  foreach(_VAR ${ARGN})
    if(NOT DEFINED "${_VAR}")
      message(SEND_ERROR "${_VAR} is not defined")
    endif()
  endforeach()
endmacro()

# iree_validate_required_arguments
#
# Validates that no arguments went unparsed or were given no values and that all
# required arguments have values. Expects to be called after
# cmake_parse_arguments and verifies that the variables it creates have been
# populated as appropriate.
function(iree_validate_required_arguments
         PREFIX
         REQUIRED_ONE_VALUE_KEYWORDS
         REQUIRED_MULTI_VALUE_KEYWORDS)
  if(DEFINED ${PREFIX}_UNPARSED_ARGUMENTS)
    message(SEND_ERROR "Unparsed argument(s): '${${PREFIX}_UNPARSED_ARGUMENTS}'")
  endif()
  if(DEFINED ${PREFIX}_KEYWORDS_MISSING_VALUES)
    message(SEND_ERROR
            "No values for field(s) '${${PREFIX}_KEYWORDS_MISSING_VALUES}'")
  endif()

  foreach(_KEYWORD IN LISTS REQUIRED_ONE_VALUE_KEYWORDS REQUIRED_MULTI_VALUE_KEYWORDS)
    if(NOT DEFINED ${PREFIX}_${_KEYWORD})
      message(SEND_ERROR "Missing required argument ${_KEYWORD}")
    endif()
  endforeach()
endfunction()
