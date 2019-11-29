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

// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --output_types=f | IreeFileCheck %s

// CHECK-LABEL: EXEC @scalar
func @scalar() -> tensor<f32> {
  %input1 = constant dense<16.0> : tensor<f32>
  %input2 = constant dense<7.0> : tensor<f32>
  %result = "xla_hlo.remainder"(%input1, %input2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=2

// CHECK-LABEL: EXEC @tensor
func @tensor() -> tensor<3xf32> {
  %input1 = constant dense<[16.0, 17.0, 18.0]> : tensor<3xf32>
  %input2 = constant dense<[7.0, 8.0, 9.0]> : tensor<3xf32>
  %result = "xla_hlo.remainder"(%input1, %input2) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  return %result : tensor<3xf32>
}
// CHECK: f32=2 1 0

// CHECK-LABEL: EXEC @negative_den
func @negative_den() -> tensor<f32> {
  %input1 = constant dense<16.0> : tensor<f32>
  %input2 = constant dense<-7.0> : tensor<f32>
  %result = "xla_hlo.remainder"(%input1, %input2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=2

// CHECK-LABEL: EXEC @negative_num
func @negative_num() -> tensor<f32> {
  %input1 = constant dense<-16.0> : tensor<f32>
  %input2 = constant dense<7.0> : tensor<f32>
  %result = "xla_hlo.remainder"(%input1, %input2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=-2
