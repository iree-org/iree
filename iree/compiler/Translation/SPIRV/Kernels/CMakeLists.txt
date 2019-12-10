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

# TODO(benvanik): iree_glsl_cc_library.
# iree_glsl_cc_library(
#   NAME
#     Kernels
#  TARGET
#    spirv
#  SRCS
#    "matmul.comp"
#    "reduce_untiled.comp"
#   PUBLIC
# )

add_custom_command(
  OUTPUT matmul.spv
  COMMAND glslangValidator -V "${CMAKE_CURRENT_SOURCE_DIR}/matmul.comp" -o matmul.spv
  DEPENDS glslangValidator
)

add_custom_command(
  OUTPUT reduce_untiled.spv
  COMMAND glslangValidator -V "${CMAKE_CURRENT_SOURCE_DIR}/reduce_untiled.comp" -o reduce_untiled.spv
  DEPENDS glslangValidator
)

add_custom_command(
  OUTPUT Kernels.h Kernels.cc
  COMMAND generate_cc_embed_data matmul.spv reduce_untiled.spv
          --output_header=Kernels.h
          --output_impl=Kernels.cc
          --identifier=Kernels
          --cpp_namespace=mlir::iree_compiler::spirv_kernels
          --flatten
  DEPENDS generate_cc_embed_data
          matmul.spv
          reduce_untiled.spv
)

add_library(iree_compiler_Translation_SPIRV_Kernels STATIC Kernels.cc)
