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
    "assign_executable_workloads.mlir"
    "fold_compatible_dispatch_regions.mlir"
    "form_streams.mlir"
    "identify_dispatch_regions.mlir"
    "identify_reduction_regions.mlir"
    "legalize_input_types.mlir"
    "materialize_and_merge_exported_reflection.mlir"
    "materialize_exported_reflection.mlir"
    "merge_exported_reflection.mlir"
    "rematerialize_dispatch_constants.mlir"
    "transformation.mlir"
  DATA
    iree::tools::IreeFileCheck
    iree::tools::iree-opt
)
