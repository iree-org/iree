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

// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --input_values="2x3xi32=[1 2 3 4 5 6]" --output_types=i | IreeFileCheck %s

// CHECK-LABEL: EXEC @pad
func @pad(%0: tensor<2x3xi32>) -> tensor<4x13xi32>
    attributes { iree.module.export } {
  %1 = constant dense<0> : tensor<i32>
  %2 = "xla_hlo.pad"(%0, %1) {edge_padding_low = dense<[0, 1]> : tensor<2xi64>, edge_padding_high = dense<[1, 5]> : tensor<2xi64>, interior_padding = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x3xi32>, tensor<i32>) -> tensor<4x13xi32>
  return %2 : tensor<4x13xi32>
}

// CHECK-NEXT: 4x13xi32=
// CHECK-SAME: [0 1 0 0 2 0 0 3 0 0 0 0 0]
// CHECK-SAME: [0 0 0 0 0 0 0 0 0 0 0 0 0]
// CHECK-SAME: [0 4 0 0 5 0 0 6 0 0 0 0 0]
// CHECK-SAME: [0 0 0 0 0 0 0 0 0 0 0 0 0]
