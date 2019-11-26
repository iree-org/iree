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

// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --output_types=i | FileCheck %s --enable-var-scope

// CHECK-LABEL: EXEC @tensor
func @tensor() -> tensor<4xi32> {
  %lhs = constant dense<[1, 6, 7, 8]> : tensor<4xi32>
  %rhs = constant dense<[5, 6, 3, 8]> : tensor<4xi32>
  %result = "xla_hlo.max"(%lhs, %rhs) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}
// CHECK: 4xi32=5 6 7 8

// -----

// CHECK-LABEL: EXEC @tensor_odd_dim
func @tensor_odd_dim() -> tensor<3xi32> {
  %lhs = constant dense<[1, 6, 7]> : tensor<3xi32>
  %rhs = constant dense<[5, 6, 3]> : tensor<3xi32>
  %result = "xla_hlo.max"(%lhs, %rhs) : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi32>
  return %result : tensor<3xi32>
}
// CHECK: 3xi32=5 6 7

// -----

// CHECK-LABEL: EXEC @scalar
func @scalar() -> tensor<i32> {
  %lhs = constant dense<1> : tensor<i32>
  %rhs = constant dense<2> : tensor<i32>
  %result = "xla_hlo.max"(%lhs, %rhs) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %result : tensor<i32>
}
// CHECK: i32=2

// -----

// CHECK-LABEL: EXEC @negative
func @negative() -> tensor<i32> {
  %lhs = constant dense<1> : tensor<i32>
  %rhs = constant dense<-2> : tensor<i32>
  %result = "xla_hlo.max"(%lhs, %rhs) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %result : tensor<i32>
}
// CHECK: i32=1

// -----

// CHECK-LABEL: EXEC @i16
func @i16() -> tensor<i16> {
  %lhs = constant dense<1> : tensor<i16>
  %rhs = constant dense<2> : tensor<i16>
  %result = "xla_hlo.max"(%lhs, %rhs) : (tensor<i16>, tensor<i16>) -> tensor<i16>
  return %result : tensor<i16>
}
// CHECK: i16=2

// -----

// CHECK-LABEL: EXEC @i64
func @i64() -> tensor<i64> {
  %lhs = constant dense<1> : tensor<i64>
  %rhs = constant dense<2> : tensor<i64>
  %result = "xla_hlo.max"(%lhs, %rhs) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  return %result : tensor<i64>
}
// CHECK: i64=2
