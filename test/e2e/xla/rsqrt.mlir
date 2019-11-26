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

// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --output_types=f | FileCheck %s --enable-var-scope

// CHECK-LABEL: EXEC @tensor
func @tensor() -> tensor<4xf32> {
  %input = constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %result = "xla_hlo.rsqrt"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}
// CHECK: 4xf32=1 0.707107 0.57735 0.5

// -----

// CHECK-LABEL: EXEC @scalar
func @scalar() -> tensor<f32> {
  %input = constant dense<16.0> : tensor<f32>
  %result = "xla_hlo.rsqrt"(%input) : (tensor<f32>) -> tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=0.25

// -----

// CHECK-LABEL: EXEC @double
func @double() -> tensor<f64> {
  %input = constant dense<16.0> : tensor<f64>
  %result = "xla_hlo.rsqrt"(%input) : (tensor<f64>) -> tensor<f64>
  return %result : tensor<f64>
}
// CHECK: f64=0.25
