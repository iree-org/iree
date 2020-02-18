# Copyright 2020 Google LLC
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

# iree_py_library()
#
# CMake function to imitate Bazel's iree_py_library rule.
#
# Parameters:
# NAME: name of target
# SRCS: List of source files for the library
# DEPS: List of other targets the test python libraries require

function(iree_py_library)

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DEPS"
    ${ARGN}
  )

  iree_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::iree::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  # Add path to each source file
  list(TRANSFORM _RULE_SRCS PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

  add_custom_target(${_NAME} ALL
    COMMAND ${CMAKE_COMMAND} -E copy "${_RULE_SRCS}" "${CMAKE_CURRENT_BINARY_DIR}/"
    DEPENDS ${_RULE_DEPS}
  )

endfunction()
