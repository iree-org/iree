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

iree_add_all_subdirs()

file(GLOB _GLOB_X_MLIR LIST_DIRECTORIES false RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} CONFIGURE_DEPENDS *.mlir)
iree_check_single_backend_test_suite(
  NAME
    check_vmla_vmla
  SRCS
    "${_GLOB_X_MLIR}"
  TARGET_BACKEND
    vmla
  DRIVER
    vmla
)

iree_check_single_backend_test_suite(
  NAME
    check_vulkan-spirv_vulkan
  SRCS
    "abs.mlir"
    "add.mlir"
    "batch_norm_inference.mlir"
    "broadcast.mlir"
    "broadcast_add.mlir"
    "broadcast_in_dim.mlir"
    "clamp.mlir"
    "compare.mlir"
    "concatenate.mlir"
    "constant.mlir"
    "convert.mlir"
    "convolution.mlir"
    "cosine.mlir"
    "divide.mlir"
    "dot.mlir"
    "exponential.mlir"
    "gather.mlir"
    "gather_concat.mlir"
    "iota.mlir"
    "log.mlir"
    "log_plus_one.mlir"
    "maximum.mlir"
    "minimum.mlir"
    "multiply.mlir"
    "negate.mlir"
    "pad.mlir"
    "reduce.mlir"
    "reduce_window.mlir"
    "remainder.mlir"
    "reshape.mlir"
    "reverse.mlir"
    "rsqrt.mlir"
    "select.mlir"
    "sine.mlir"
    "slice.mlir"
    "sqrt.mlir"
    "subtract.mlir"
    "tanh.mlir"
    "torch_index_select.mlir"
    "transpose.mlir"
    "while.mlir"
  TARGET_BACKEND
    vulkan-spirv
  DRIVER
    vulkan
)

iree_check_single_backend_test_suite(
  NAME
    check_llvm-ir_llvm
  SRCS
    "abs.mlir"
    "add.mlir"
    "batch_norm_inference.mlir"
    "broadcast.mlir"
    "broadcast_add.mlir"
    "broadcast_in_dim.mlir"
    "clamp.mlir"
    "compare.mlir"
    "concatenate.mlir"
    "constant.mlir"
    "convolution.mlir"
    "cosine.mlir"
    "divide.mlir"
    "dot.mlir"
    "exponential.mlir"
    "gather.mlir"
    "gather_concat.mlir"
    "iota.mlir"
    "log.mlir"
    "log_plus_one.mlir"
    "maximum.mlir"
    "minimum.mlir"
    "multiply.mlir"
    "negate.mlir"
    "pad.mlir"
    "reduce.mlir"
    "reduce_window.mlir"
    "remainder.mlir"
    "reshape.mlir"
    "reverse.mlir"
    "rsqrt.mlir"
    "select.mlir"
    "sine.mlir"
    "slice.mlir"
    "sqrt.mlir"
    "subtract.mlir"
    "tanh.mlir"
    "torch_index_select.mlir"
    "transpose.mlir"
    "while.mlir"
  TARGET_BACKEND
    llvm-ir
  DRIVER
    llvm
)
