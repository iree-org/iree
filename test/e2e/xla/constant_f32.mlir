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

// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s | FileCheck %s --enable-var-scope --dump-input=fail
// RUN: iree-run-mlir --target_backends=vulkan-spirv %s | FileCheck %s --enable-var-scope --dump-input=fail

// -----

// CHECK-LABEL: EXEC @xla_constant_f32
func @xla_constant_f32 () -> (tensor<2x2x3xf32>, tensor<2x2x3xf32>) {
  %0 = xla_hlo.constant dense<[[[1.1e0, 2.1e0, 3.1e0], [4.1e0, 5.1e0, 6.1e0]], [[7.1e0, 8.1e0, 9.1e0], [10.1e0, 11.1e0, 12.1e0]]]> : tensor<2x2x3xf32>
  %1 = xla_hlo.constant dense<1.1e0> : tensor<2x2x3xf32>
  return %0, %1: tensor<2x2x3xf32>, tensor<2x2x3xf32>
}
// CHECK: 2x2x3xf32={{\[}}[1.1 2.1 3.1][4.1 5.1 6.1]]{{\[}}[7.1 8.1 9.1][10.1 11.1 12.1]]
// CHECK-NEXT: 2x2x3xf32={{\[}}[1.1 1.1 1.1][1.1 1.1 1.1]]{{\[}}[1.1 1.1 1.1][1.1 1.1 1.1]]
