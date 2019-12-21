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
  OUTPUT conv2d_nhwc.spv
  COMMAND glslangValidator -V "${CMAKE_CURRENT_SOURCE_DIR}/conv2d_nhwc.comp" -o conv2d_nhwc.spv
  DEPENDS glslangValidator
)

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

iree_cc_embed_data(
  NAME
    Kernels
  GENERATED_SRCS
    "conv2d_nhwc.spv"
    "matmul.spv"
    "reduce_untiled.spv"
  CC_FILE_OUTPUT
    "Kernels.cc"
  H_FILE_OUTPUT
    "Kernels.h"
  CPP_NAMESPACE
    "mlir::iree_compiler::spirv_kernels"
  FLATTEN
)
