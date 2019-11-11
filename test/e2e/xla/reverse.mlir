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

// RUN: iree-run-mlir --target_backends=interpreter-bytecode --input_values="2x3xf32= 1.0 2.0 3.0 4.0 5.0 6.0" %s | FileCheck %s --dump-input=fail
// RUN: iree-run-mlir --target_backends=vulkan-spirv --input_values="2x3xf32= 1.0 2.0 3.0 4.0 5.0 6.0" %s | FileCheck %s --dump-input=fail

// -----

// CHECK-LABEL: EXEC @xla_reverse
func @xla_reverse (%t1: tensor<2x3xf32>) -> (tensor<2x3xf32>, tensor <2x3xf32>, tensor <2x3xf32>) {
  // TODO(b/143512073): Make %t1 as constant instead of argument. Currently
  // vulkan-spirv does not handle index propagation for xla_hlo.constant.
  // %t1 = xla_hlo.constant dense<[[1.0e0, 2.0e0, 3.0e0], [4.0e0, 5.0e0, 6.0e0]]> : tensor<2x3xf32>
  %0 = "xla_hlo.reverse"(%t1) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "xla_hlo.reverse"(%t1) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %2 = "xla_hlo.reverse"(%t1) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0, %1, %2: tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>
}
// CHECK: 2x3xf32=[4 5 6][1 2 3]
// CHECK-NEXT: 2x3xf32=[3 2 1][6 5 4]
// CHECK_NEXT: 2x3xf32=[6 5 4][3 2 1]
