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

// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --input_values="4xi8=[1 0 200 0]" | FileCheck %s --enable-var-scope
// CHECK-LABEL: EXEC @select
func @select(%cond : tensor<4xi1>) -> tensor<4xf32> {
  %lhs = constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %rhs = constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>
  %result = "xla_hlo.select"(%cond, %lhs, %rhs) : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}
// CHECK: 4xf32=1 6 3 8
