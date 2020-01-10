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

# iree_hlsl_vulkan()
#
# CMake function to imitate Bazel's iree_hlsl_vulkan rule and hlsl_vulkan rule
#
# Parameters:
# NAME: Name of spv file to create (without file name extension).
# SRC: GLSL source file to translate into a SPIR-V binary.

function(iree_hlsl_vulkan)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC"
    ""
    ${ARGN}
  )

  iree_glslang(
    NAME
      ${_RULE_NAME}
    SRC
      ${_RULE_SRC}
    MODE
      "hlsl"
    TARGET
      "vulkan"
  )

endfunction()
