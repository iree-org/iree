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

// RUN: iree-run-mlir --target_backends=interpreter-bytecode --output_types=i,i %s | FileCheck %s --enable-var-scope --dump-input=fail
// RUN: iree-run-mlir --target_backends=vulkan-spirv --output_types=i,i %s | FileCheck %s --enable-var-scope --dump-input=fail

// -----

// CHECK-LABEL: EXEC @xla_constant_i32
func @xla_constant_i32 () -> (tensor<2x2x3xi32>, tensor<2x2x3xi32>) {
  %0 = xla_hlo.constant dense<[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]> : tensor<2x2x3xi32>
  %1 = xla_hlo.constant dense<1> : tensor<2x2x3xi32>
  return %0, %1: tensor<2x2x3xi32>, tensor<2x2x3xi32>
}
// CHECK: 2x2x3xi32={{\[}}[1 2 3][4 5 6]]{{\[}}[7 8 9][10 11 12]]
// CHECK-NEXT: 2x2x3xi32={{\[}}[1 1 1][1 1 1]]{{\[}}[1 1 1][1 1 1]]
