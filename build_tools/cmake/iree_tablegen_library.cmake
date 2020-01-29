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

  if(NOT _RULE_TESTONLY OR IREE_BUILD_TESTS)
    # Prefix the library with the package name, so we get: iree_package_name
    iree_package_name(_PACKAGE_NAME)
    set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

    if(${_RULE_TBLGEN} MATCHES "IREE")
      set(_TBLGEN "IREE")
    else()
      set(_TBLGEN "MLIR")
    endif()

    set(LLVM_TARGET_DEFINITIONS ${_RULE_TD_FILE})
    set(_INCLUDE_DIRS ${IREE_COMMON_INCLUDE_DIRS})
    list(APPEND _INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
    list(TRANSFORM _INCLUDE_DIRS PREPEND "-I")
    set(_OUTPUTS)
    while(_RULE_OUTS)
      list(GET _RULE_OUTS 0 _COMMAND)
      list(REMOVE_AT _RULE_OUTS 0)
      list(GET _RULE_OUTS 0 _FILE)
      list(REMOVE_AT _RULE_OUTS 0)
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
  endif()
endfunction()
