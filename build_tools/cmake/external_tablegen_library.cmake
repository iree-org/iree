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

# external_tablegen_library()
#
# Runs ${TBLGEN} to produce some artifacts.
function(external_tablegen_library)
  cmake_parse_arguments(
    _RULE
    "TESTONLY"
    "PACKAGE;NAME;ROOT;TBLGEN"
    "SRCS;OUTS"
    ${ARGN}
  )

  if(_RULE_TESTONLY AND NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Prefix the library with the package name.
  string(REPLACE "::" "_" _PACKAGE_NAME ${_RULE_PACKAGE})
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  # Prefix source paths with the root.
  list(TRANSFORM _RULE_SRCS PREPEND ${_RULE_ROOT})

  set(LLVM_TARGET_DEFINITIONS ${_RULE_SRCS})
  set(_INCLUDE_DIRS ${IREE_COMMON_INCLUDE_DIRS})
  list(APPEND _INCLUDE_DIRS ${_RULE_ROOT})
  list(TRANSFORM _INCLUDE_DIRS PREPEND "-I")
  set(_OUTPUTS)
  while(_RULE_OUTS)
    list(GET _RULE_OUTS 0 _COMMAND)
    list(REMOVE_AT _RULE_OUTS 0)
    list(GET _RULE_OUTS 0 _FILE)
    list(REMOVE_AT _RULE_OUTS 0)
    tablegen(${_RULE_TBLGEN} ${_FILE} ${_COMMAND} ${_INCLUDE_DIRS})
    list(APPEND _OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/${_FILE})
  endwhile()
  add_custom_target(${_NAME}_target DEPENDS ${_OUTPUTS})
  set_target_properties(${_NAME}_target PROPERTIES FOLDER "Tablegenning")

  add_library(${_NAME} INTERFACE)
  add_dependencies(${_NAME} ${_NAME}_target)

  add_library(${_RULE_PACKAGE}::${_RULE_NAME} ALIAS ${_NAME})
endfunction()
