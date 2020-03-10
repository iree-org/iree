// Copyright 2020 Google LLC
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

// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s | IreeFileCheck %s

// CHECK-LABEL: EXEC @scalar
func @scalar() -> (i32, i32) {
  %result = iree.unfoldable_constant 42 : i32
  return %result, %result : i32, i32
}
// CHECK-COUNT-2: i32=42

// -----
// CHECK-LABEL: EXEC @rank0tensor
func @rank0tensor() -> (tensor<f32>, tensor<f32>) {
  %res = iree.unfoldable_constant dense<42.0> : tensor<f32>
  return %res, %res : tensor<f32>, tensor<f32>
}
// CHECK-COUNT-2: f32=42

// -----
// CHECK-LABEL: EXEC @tensor
func @tensor() -> (tensor<2x2xf32>, tensor<2x2xf32>) {
  %res = iree.unfoldable_constant
      dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  return %res, %res : tensor<2x2xf32>, tensor<2x2xf32>
}
// CHECK-COUNT-2: 2x2xf32=[1 2][3 4]

// -----
// CHECK-LABEL: EXEC @many_tensor
func @many_tensor() -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>,
                        tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) {
  %res = iree.unfoldable_constant
      dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  return %res, %res, %res, %res, %res, %res :
        tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>,
        tensor<2x2xf32>, tensor<2x2xf32>
}
// CHECK-COUNT-6: 2x2xf32=[1 2][3 4]
