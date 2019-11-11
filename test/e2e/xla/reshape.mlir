// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --input_values="12xf32=[1 2 3 4 5 6 7 8 9 10 11 12]" | FileCheck %s --dump-input=fail
// RUN: iree-run-mlir --target_backends=vulkan-spirv %s --input_values="12xf32=[1 2 3 4 5 6 7 8 9 10 11 12]" | FileCheck %s --dump-input=fail

// CHECK-LABEL: EXEC @reshape_1D_2D
func @reshape_1D_2D(%arg : tensor<12xf32>) -> tensor<3x4xf32> {
  %result = "xla_hlo.reshape"(%arg) : (tensor<12xf32>) -> tensor<3x4xf32>
  return %result : tensor<3x4xf32>
}
// CHECK: 3x4xf32=[1 2 3 4][5 6 7 8][9 10 11 12]

// CHECK-LABEL: EXEC @reshape_1D_3D
func @reshape_1D_3D(%arg : tensor<12xf32>) -> tensor<2x2x3xf32> {
  %result = "xla_hlo.reshape"(%arg) : (tensor<12xf32>) -> tensor<2x2x3xf32>
  return %result : tensor<2x2x3xf32>
}
// CHECK 2x2x3xf32=\[[1 2 3][4 5 6]]\[[7 8 9][10 11 12]]
