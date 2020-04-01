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

# iree_spirv_kernel_cc_library()
#
# CMake function to imitate Bazel's spirv_kernel_cc_library rule.
#
# Parameters:
# NAME: Name of target (see Note).
# SRCS: List of compute shader source files.
# PUBLIC: Add this so that this library will be exported under ${PACKAGE}::
# TESTONLY: When added, this target will only be built if user passes
#    -DIREE_BUILD_TESTS=ON to CMake.
#
# Note:
# By default, iree_spirv_kernel_cc_library will always create a library named ${NAME},
# and alias target iree::${NAME}. The iree:: form should always be used.
# This is to reduce namespace pollution.

function(iree_spirv_kernel_cc_library)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;TESTONLY"
    "NAME"
    "SRCS"
    ${ARGN}
  )

  if(NOT _RULE_TESTONLY OR IREE_BUILD_TESTS)
    set(_SPV_FILES)
    foreach(_SRC ${_RULE_SRCS})
      get_filename_component(_SPV_NAME ${_SRC} NAME_WE)
      list(APPEND _SPV_FILES "${_SPV_NAME}.spv")

      iree_glsl_vulkan(
        NAME
          "${_SPV_NAME}"
        SRC
          "${_SRC}"
      )
    endforeach(_SRC)

    if(_RULE_TESTONLY)
      set(_TESTONLY_ARG "TESTONLY")
    endif()
    if(_RULE_PUBLIC)
      set(_PUBLIC_ARG "PUBLIC")
    endif()

    iree_cc_embed_data(
      NAME
        "${_RULE_NAME}"
      GENERATED_SRCS
        "${_SPV_FILES}"
      CC_FILE_OUTPUT
        "${_RULE_NAME}.cc"
      H_FILE_OUTPUT
        "${_RULE_NAME}.h"
      CPP_NAMESPACE
        "mlir::iree_compiler::spirv_kernels"
      FLATTEN
      "${_PUBLIC_ARG}"
      "${_TESTONLY_ARG}"
    )
  endif()
endfunction()
