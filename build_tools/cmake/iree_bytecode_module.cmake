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

# iree_bytecode_module()
#
# CMake function to imitate Bazel's iree_bytecode_module rule.
#
# Parameters:
# NAME: Name of target (see Note).
# SRC: Source file to compile into a bytecode module.
# TRANSLATION: Translation option to pass to the translation tool (string).
# TRANSLATION_TOOL: Translation tool to invoke (CMake target).
# CC_NAMESPACE: Wraps everything in a C++ namespace.
# PUBLIC: Add this so that this library will be exported under ${PACKAGE}::
# Also in IDE, target will appear in ${PACKAGE} folder while non PUBLIC will be
# in ${PACKAGE}/internal.
# TESTONLY: When added, this target will only be built if user passes
#    -DIREE_BUILD_TESTS=ON to CMake.
#
# Note:
# By default, iree_bytecode_module will create a library named ${NAME}_cc,
# and alias target iree::${NAME}_cc. The iree:: form should always be used.
# This is to reduce namespace pollution.
function(iree_bytecode_module)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY"
    "NAME;SRC;TRANSLATION;TRANSLATION_TOOL;CC_NAMESPACE"
    ""
    ${ARGN}
  )

  if(NOT _RULE_TESTONLY OR IREE_BUILD_TESTS)
    # Set defaults for TRANSLATION and TRANSLATION_TOOL
    if(DEFINED _RULE_TRANSLATION)
      set(_TRANSLATION ${_RULE_TRANSLATION})
    else()
      set(_TRANSLATION "-mlir-to-iree-module")
    endif()
    if(DEFINED _RULE_TRANSLATION_TOOL)
      set(_TRANSLATION_TOOL ${_RULE_TRANSLATION_TOOL})
    else()
      set(_TRANSLATION_TOOL "iree_tools_iree-translate")
    endif()

    # Resolve the executable binary path from the target name.
    set(_TRANSLATION_TOOL_EXECUTABLE $<TARGET_FILE:${_TRANSLATION_TOOL}>)

    set(_ARGS "${_TRANSLATION}")
    list(APPEND _ARGS "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRC}")
    list(APPEND _ARGS "-o")
    list(APPEND _ARGS "${_RULE_NAME}.module")

    add_custom_command(
      OUTPUT "${_RULE_NAME}.module"
      COMMAND ${_TRANSLATION_TOOL_EXECUTABLE} ${_ARGS}
      DEPENDS ${_TRANSLATION_TOOL}
    )

    if(DEFINED _RULE_CC_NAMESPACE)
      iree_cc_embed_data(
        NAME
          "${_RULE_NAME}_cc"
        PUBLIC
          "${_RULE_PUBLIC}"
        TESTONLY
          "${_RULE_TESTONLY}"
        IDENTIFIER
          "${_RULE_NAME}"
        GENERATED_SRCS
          "${_RULE_NAME}.module"
        CC_FILE_OUTPUT
          "${_RULE_NAME}.cc"
        H_FILE_OUTPUT
          "${_RULE_NAME}.h"
        CPP_NAMESPACE
          "${_RULE_CC_NAMESPACE}"
        FLATTEN
      )
    endif()
  endif()
endfunction()
