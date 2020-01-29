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
    "arithmetic_folding.mlir"
    "arithmetic_ops.mlir"
    "assignment_folding.mlir"
    "assignment_ops.mlir"
    "comparison_folding.mlir"
    "comparison_ops.mlir"
    "const_folding.mlir"
    "const_ops.mlir"
    "control_flow_folding.mlir"
    "control_flow_ops.mlir"
    "conversion_folding.mlir"
    "conversion_ops.mlir"
    "debug_folding.mlir"
    "debug_ops.mlir"
    "global_folding.mlir"
    "global_ops.mlir"
    "structural_ops.mlir"
  DATA
    iree::tools::IreeFileCheck
    iree::tools::iree-opt
)
