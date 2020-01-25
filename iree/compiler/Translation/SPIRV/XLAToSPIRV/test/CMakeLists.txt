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

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "adjust_integer_width.mlir"
    "arithmetic_ops.mlir"
    "broadcast.mlir"
    "broadcast_in_dim.mlir"
    "concatenate.mlir"
    "constant.mlir"
    "convert.mlir"
    "copy.mlir"
    "exp_test.mlir"
    "extract_element.mlir"
    "gather.mlir"
    "max.mlir"
    "pad.mlir"
    "reshape.mlir"
    "reshape_dropdims.mlir"
    "reverse.mlir"
    "select.mlir"
    "slice.mlir"
    "store_reduce.mlir"
    "transpose_add.mlir"
  DATA
    iree::tools::IreeFileCheck
    iree::tools::iree-opt
)
