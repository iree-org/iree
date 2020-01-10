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

# iree_glslang()
#
# CMake function to imitate Bazel's _glslang rule.
#
# Parameters:
# NAME: Name of spv file to create (without file name extension).
# SRC: Source file.
# MODE: Defines the type of input: either "hlsl" or "glsl".
# TARGET: Target to create the SPIR-V binary for: either "vulkan" or "opengl".

function(iree_glslang)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC;MODE;TARGET"
    ""
    ${ARGN}
  )

  if(_RULE_MODE STREQUAL "glsl")
    set(_MODE "")
  elseif(_RULE_MODE STREQUAL "hlsl")
    set(_MODE "-D")
  else()
    message(FATAL_ERROR "Illegal mode ${_RULE_MODE}")
  endif()

  if(_RULE_TARGET STREQUAL "opengl")
    set(_TARGET "-G")
  elseif(_RULE_TARGET STREQUAL "vulkan")
    set(_TARGET "-V")
  else()
    message(FATAL_ERROR "Illegal target ${_RULE_TARGET}")
  endif()

  set(_ARGS "${_MODE}")
  list(APPEND _ARGS "${_TARGET}")
  list(APPEND _ARGS "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRC}")
  list(APPEND _ARGS "-o")
  list(APPEND _ARGS "${_RULE_NAME}.spv")

  add_custom_command(
    OUTPUT "${_RULE_NAME}.spv"
    COMMAND glslangValidator ${_ARGS}
    DEPENDS glslangValidator
  )

endfunction()
