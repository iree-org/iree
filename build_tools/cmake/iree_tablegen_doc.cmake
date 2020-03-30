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

# iree_tablegen_doc()
#
# Runs iree-tablegen to produce documentation. For TableGen'ing others,
# please use iree_tablegen_library().
#
# One-value parameters:
# * NAME: base name of the target. The real target name is mangled from this
#         given name with the package name
# * TBLGEN: the base project to pass to TableGen
#
# Multi-value parameters:
# * TD_FILE: Input .td files
# * OUTS: TableGen generator commands and their outputs in the format of
#         `-gen-<something> <output-file-name>`. Note that the generator
#         commands should only be for documentation.
function(iree_tablegen_doc)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TBLGEN"
    "TD_FILE;OUTS"
    ${ARGN}
  )

  if(IREE_BUILD_DOCS)
    # Prefix the library with the package name, so we get: iree_package_name
    iree_package_name(_PACKAGE_NAME)
    set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

    if(${_RULE_TBLGEN} MATCHES "IREE")
      set(_TBLGEN "IREE")
    else()
      set(_TBLGEN "MLIR")
    endif()


    set(_INCLUDE_DIRS ${IREE_COMMON_INCLUDE_DIRS})
    list(APPEND _INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
    list(TRANSFORM _INCLUDE_DIRS PREPEND "-I")

    set(_INPUTS ${_RULE_TD_FILE})
    set(LLVM_TARGET_DEFINITIONS ${_INPUTS})

    set(_OUTPUTS)
    while(_RULE_OUTS)
      list(GET _RULE_OUTS 0 _COMMAND)
      list(REMOVE_AT _RULE_OUTS 0)
      list(GET _RULE_OUTS 0 _OUTPUT)
      list(REMOVE_AT _RULE_OUTS 0)

      # TableGen this output with the given command.
      tablegen(${_TBLGEN} ${_OUTPUT} ${_COMMAND} ${_INCLUDE_DIRS})
      list(APPEND _OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/${_OUTPUT})
    endwhile()

    # Put all dialect docs at one place.
    set(_DOC_DIR ${PROJECT_BINARY_DIR}/doc/Dialects/)
    # Set a target to drive copy.
    add_custom_target(${_NAME}_target
              ${CMAKE_COMMAND} -E make_directory ${_DOC_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy ${_OUTPUTS} ${_DOC_DIR}
      DEPENDS ${_OUTPUTS})
    set_target_properties(${_NAME}_target PROPERTIES FOLDER "Tablegenning")

    # Register this dialect doc to iree-doc.
    add_dependencies(iree-doc ${_NAME}_target)
  endif()
endfunction()
