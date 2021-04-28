# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(CMakeParseArguments)

# iree_cc_binary()
#
# CMake function to imitate Bazel's cc_binary rule.
#
# Parameters:
# NAME: name of target (see Usage below)
# OUT: OUTPUT_NAME for the target. Defaults to NAME.
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
# By default, iree_cc_binary will always create a binary named iree_${NAME}.
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
    "HOSTONLY;TESTONLY"
    "NAME;OUT"
    "SRCS;COPTS;DEFINES;LINKOPTS;DATA;DEPS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Prefix the library with the package name, so we get: iree_package_name
  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  add_executable(${_NAME} "")
  add_executable(${_RULE_NAME} ALIAS ${_NAME})
  if(_RULE_OUT)
    set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_OUT}")
  else()
    set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_NAME}")
  endif()
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
  iree_package_ns(_PACKAGE_NS)
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  target_link_libraries(${_NAME}
    PUBLIC
      ${_RULE_DEPS}
  )
  iree_add_data_dependencies(NAME ${_NAME} DATA ${_RULE_DATA})

  # Add all IREE targets to a folder in the IDE for organization.
  set_property(TARGET ${_NAME} PROPERTY FOLDER ${IREE_IDE_FOLDER}/binaries)

  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${IREE_CXX_STANDARD})
  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

  install(TARGETS ${_NAME}
          RENAME ${_RULE_NAME}
          COMPONENT ${_RULE_NAME}
          RUNTIME DESTINATION bin)
endfunction()
