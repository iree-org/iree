# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# iree_cc_binary()
#
# CMake function to imitate Bazel's cc_binary rule.
#
# Parameters:
# NAME: name of target (see Usage below)
# SRCS: List of source files for the binary
# DATA: List of other targets and files required for this binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
# TESTONLY: for testing; won't compile when tests are disabled
# HOSTONLY: host only; compile using host toolchain when cross-compiling
#
# Note:
# iree_cc_binary will create a binary called ${PACKAGE_NAME}_${NAME}, e.g.
# iree_base_foo with two alias (readonly) targets, a qualified
# ${PACKAGE_NS}::${NAME} and an unqualified ${NAME}. Thus NAME must be globally
# unique in the project.
#
# Usage:
# iree_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
#   PUBLIC
# )
#
# iree_cc_binary(
#   NAME
#     awesome_tool
#   SRCS
#     "awesome-tool-main.cc"
#   DEPS
#     iree::awesome
# )
function(iree_cc_binary)
  cmake_parse_arguments(
    _RULE
    "EXCLUDE_FROM_ALL;HOSTONLY;TESTONLY"
    "NAME"
    "SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Prefix the library with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  iree_package_ns(_PACKAGE_NS)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  add_executable(${_NAME} "")
  # Alias the iree_package_name binary to iree::package::name.
  # This lets us more clearly map to Bazel and makes it possible to
  # disambiguate the underscores in paths vs. the separators.
  add_executable(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})

  # If the binary name matches the package then treat it as a default. For
  # example, foo/bar/ library 'bar' would end up as 'foo::bar'. This isn't
  # likely to be common for binaries, but is consistent with the behavior for
  # libraries and in Bazel.
  iree_package_dir(_PACKAGE_DIR)
  if(${_RULE_NAME} STREQUAL ${_PACKAGE_DIR})
    add_executable(${_PACKAGE_NS} ALIAS ${_NAME})
  endif()

  # Finally, since we have so few binaries and we also want to support
  # installing from a separate host build, binaries get an unqualified global
  # alias. This means binary names must be unique across the whole project.
  # (We could consider making this configurable).
  add_executable(${_RULE_NAME} ALIAS ${_NAME})

  set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_NAME}")
  if(_RULE_SRCS)
    target_sources(${_NAME}
      PRIVATE
        ${_RULE_SRCS}
    )
  else()
    set(_DUMMY_SRC "${CMAKE_CURRENT_BINARY_DIR}/${_NAME}_dummy.cc")
    file(WRITE ${_DUMMY_SRC} "")
    target_sources(${_NAME}
      PRIVATE
        ${_DUMMY_SRC}
    )
  endif()
  target_include_directories(${_NAME} SYSTEM
    PUBLIC
      "$<BUILD_INTERFACE:${IREE_SOURCE_DIR}>"
      "$<BUILD_INTERFACE:${IREE_BINARY_DIR}>"
  )
  target_compile_definitions(${_NAME}
    PUBLIC
      ${_RULE_DEFINES}
  )
  target_compile_options(${_NAME}
    PRIVATE
      ${IREE_DEFAULT_COPTS}
      ${_RULE_COPTS}
  )
  target_link_options(${_NAME}
    PRIVATE
      ${IREE_DEFAULT_LINKOPTS}
      ${_RULE_LINKOPTS}
  )

  # Replace dependencies passed by ::name with iree::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  # Implicit deps.
  if(IREE_IMPLICIT_DEFS_CC_DEPS)
    list(APPEND _RULE_DEPS ${IREE_IMPLICIT_DEFS_CC_DEPS})
  endif()

  target_link_libraries(${_NAME}
    PUBLIC
      ${_RULE_DEPS}
  )
  iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})

  # Add all IREE targets to a folder in the IDE for organization.
  set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/binaries)

  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${IREE_CXX_STANDARD})
  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

  if(_RULE_EXCLUDE_FROM_ALL)
    set_property(TARGET ${_NAME} PROPERTY EXCLUDE_FROM_ALL ON)
    install(TARGETS ${_NAME}
            RENAME ${_RULE_NAME}
            COMPONENT ${_RULE_NAME}
            RUNTIME DESTINATION bin
            EXCLUDE_FROM_ALL)
  else()
    install(TARGETS ${_NAME}
      RENAME ${_RULE_NAME}
      COMPONENT ${_RULE_NAME}
      RUNTIME DESTINATION bin)
  endif()
endfunction()
