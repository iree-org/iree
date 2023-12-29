# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# external_cc_library()
#
# CMake function to imitate Bazel's cc_library rule.
# This is used for external libraries (from third_party, etc) that don't live
# in the IREE namespace.
#
# Parameters:
# PACKAGE: Name of the package (overrides actual path)
# NAME: Name of target (see Note)
# ROOT: Path to the source root where files are found
# HDRS: List of public header files for the library
# SRCS: List of source files for the library
# DATA: List of other targets and files required for this binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# INCLUDES: Include directories to add to dependencies
# LINKOPTS: List of link options
# PUBLIC: Add this so that this library will be exported under ${PACKAGE}::
# Also in IDE, target will appear in ${PACKAGE} folder while non PUBLIC will be
# in ${PACKAGE}/internal.
# TESTONLY: When added, this target will only be built if user passes
#    -DIREE_BUILD_TESTS=ON to CMake.
#
# Note:
# By default, external_cc_library will always create a library named
# ${PACKAGE}_${NAME}, and alias target ${PACKAGE}::${NAME}. The ${PACKAGE}::
# form should always be used. This is to reduce namespace pollution.
#
# external_cc_library(
#   PACKAGE
#     some_external_thing
#   NAME
#     awesome
#   ROOT
#     "third_party/foo"
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
# external_cc_library(
#   PACKAGE
#     some_external_thing
#   NAME
#     fantastic_lib
#   ROOT
#     "third_party/foo"
#   SRCS
#     "b.cc"
#   DEPS
#     some_external_thing::awesome # not "awesome" !
#   PUBLIC
# )
#
# iree_cc_library(
#   NAME
#     main_lib
#   ...
#   DEPS
#     some_external_thing::fantastic_lib
# )
function(external_cc_library)
  cmake_parse_arguments(_RULE
    "PUBLIC;TESTONLY"
    "PACKAGE;NAME;ROOT"
    "HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS;INCLUDES"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Prefix the library with the package name.
  string(REPLACE "::" "_" _PACKAGE_NAME ${_RULE_PACKAGE})
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  # Prefix paths with the root.
  list(TRANSFORM _RULE_HDRS PREPEND ${_RULE_ROOT})
  list(TRANSFORM _RULE_SRCS PREPEND ${_RULE_ROOT})

  # Check if this is a header-only library.
  # Note that as of February 2019, many popular OS's (for example, Ubuntu
  # 16.04 LTS) only come with cmake 3.5 by default.  For this reason, we can't
  # use list(FILTER...)
  set(_CC_SRCS "${_RULE_SRCS}")
  foreach(_SRC_FILE IN LISTS _CC_SRCS)
    if(${_SRC_FILE} MATCHES ".*\\.(h|inc)$")
      list(REMOVE_ITEM _CC_SRCS "${_SRC_FILE}")
    endif()
  endforeach()
  if("${_CC_SRCS}" STREQUAL "")
    set(_RULE_IS_INTERFACE 1)
  else()
    set(_RULE_IS_INTERFACE 0)
  endif()

  if(NOT _RULE_IS_INTERFACE)
    add_library(${_NAME} STATIC "")
    target_sources(${_NAME}
      PRIVATE
        ${_RULE_SRCS}
        ${_RULE_HDRS}
    )
    target_include_directories(${_NAME}
      PUBLIC
        "$<BUILD_INTERFACE:${IREE_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${IREE_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${_RULE_INCLUDES}>"
    )
    target_compile_options(${_NAME}
      PRIVATE
        ${_RULE_COPTS}
        ${IREE_DEFAULT_COPTS}
    )
    target_link_options(${_NAME}
      PRIVATE
        ${IREE_DEFAULT_LINKOPTS}
        ${_RULE_LINKOPTS}
    )
    target_link_libraries(${_NAME}
      PUBLIC
        ${_RULE_DEPS}
    )
    target_compile_definitions(${_NAME}
      PUBLIC
        ${_RULE_DEFINES}
    )
    iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})

    # Add all external targets to a a folder in the IDE for organization.
    if(_RULE_PUBLIC)
      set_property(TARGET ${_NAME} PROPERTY FOLDER third_party)
    elseif(_RULE_TESTONLY)
      set_property(TARGET ${_NAME} PROPERTY FOLDER third_party/test)
    else()
      set_property(TARGET ${_NAME} PROPERTY FOLDER third_party/internal)
    endif()

    # INTERFACE libraries can't have the CXX_STANDARD property set
    set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${IREE_CXX_STANDARD})
    set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
  else()
    # Generating header-only library
    add_library(${_NAME} INTERFACE)
    target_include_directories(${_NAME}
      INTERFACE
        "$<BUILD_INTERFACE:${IREE_SOURCE_DIR}>"
        "$<BUILD_INTERFACE:${IREE_BINARY_DIR}>"
        "$<BUILD_INTERFACE:${_RULE_INCLUDES}>"
    )
    target_link_libraries(${_NAME}
      INTERFACE
        ${_RULE_DEPS}
    )
    iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})
    target_compile_definitions(${_NAME}
      INTERFACE
        ${_RULE_DEFINES}
    )
  endif()

  iree_install_targets(
    TARGETS ${_NAME}
    HDRS ${_RULE_HDRS}
  )    

  iree_add_alias_library(${_RULE_PACKAGE}::${_RULE_NAME} ${_NAME})
  # If the library name matches the final component of the package then treat it
  # as a default. For example, 'foo::bar' library 'bar' would end up as
  # 'foo::bar'.
  string(REGEX REPLACE "^.*::" "" _PACKAGE_DIR ${_RULE_PACKAGE})
  if(${_PACKAGE_DIR} STREQUAL ${_RULE_NAME})
    iree_add_alias_library(${_RULE_PACKAGE} ${_NAME})
  endif()
endfunction()
