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

// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --output_types=i | FileCheck %s

// CHECK-LABEL: EXEC @compare_tensor
func @compare_tensor() -> tensor<4xi1> {
  %lhs = constant dense<[1, 2, 7, 4]> : tensor<4xi32>
  %rhs = constant dense<[5, 2, 3, 4]> : tensor<4xi32>
  %result = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %result : tensor<4xi1>
}
// CHECK: 4xi8=0 1 0 1

// -----

// CHECK-LABEL: EXEC @compare_scalar
func @compare_scalar() -> tensor<i1> {
  %lhs = constant dense<1> : tensor<i32>
  %rhs = constant dense<5> : tensor<i32>
  %result = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %result : tensor<i1>
}
// CHECK: i8=0

// -----

// CHECK-LABEL: EXEC @compare_i8
func @compare_i8() -> tensor<i1> {
  %lhs = constant dense<1> : tensor<i8>
  %rhs = constant dense<5> : tensor<i8>
  %result = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<i8>, tensor<i8>) -> tensor<i1>
  return %result : tensor<i1>
}
// CHECK: i8=0

// -----

// CHECK-LABEL: EXEC @compare_i16
func @compare_i16() -> tensor<i1> {
  %lhs = constant dense<1> : tensor<i16>
  %rhs = constant dense<5> : tensor<i16>
  %result = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<i16>, tensor<i16>) -> tensor<i1>
  return %result : tensor<i1>
}
// CHECK: i8=0

// -----

// CHECK-LABEL: EXEC @compare_i32
func @compare_i32() -> tensor<i1> {
  %lhs = constant dense<1> : tensor<i32>
  %rhs = constant dense<5> : tensor<i32>
  %result = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %result : tensor<i1>
}
// CHECK: i8=0

// -----

// CHECK-LABEL: EXEC @compare_i64
func @compare_i64() -> tensor<i1> {
  %lhs = constant dense<1> : tensor<i64>
  %rhs = constant dense<5> : tensor<i64>
  %result = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  return %result : tensor<i1>
}
// CHECK: i8=0

// -----

// CHECK-LABEL: EXEC @compare_f32
func @compare_f32() -> tensor<i1> {
  %lhs = constant dense<1.0> : tensor<f32>
  %rhs = constant dense<5.0> : tensor<f32>
  %result = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  return %result : tensor<i1>
}
// CHECK: i8=0

// -----

// CHECK-LABEL: EXEC @compare_f64
func @compare_f64() -> tensor<i1> {
  %lhs = constant dense<1.0> : tensor<f64>
  %rhs = constant dense<5.0> : tensor<f64>
  %result = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<f64>, tensor<f64>) -> tensor<i1>
  return %result : tensor<i1>
}
// CHECK: i8=0
// -----

// CHECK-LABEL: EXEC @compare_tensor_odd_length
func @compare_tensor_odd_length() -> tensor<3xi1> {
  %lhs = constant dense<[1, 2, 7]> : tensor<3xi32>
  %rhs = constant dense<[5, 2, 3]> : tensor<3xi32>
  %result = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  return %result : tensor<3xi1>
}
// CHECK: 3xi8=0 1 0

// -----

// CHECK-LABEL: EXEC @compare_eq
func @compare_eq() -> tensor<4xi1> {
  %lhs = constant dense<[1, 2, 7, 4]> : tensor<4xi32>
  %rhs = constant dense<[5, 2, 3, 4]> : tensor<4xi32>
  %result = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %result : tensor<4xi1>
}
// CHECK: 4xi8=0 1 0 1

// -----

// CHECK-LABEL: EXEC @compare_ne
func @compare_ne() -> tensor<4xi1> {
  %lhs = constant dense<[1, 2, 7, 4]> : tensor<4xi32>
  %rhs = constant dense<[5, 2, 3, 4]> : tensor<4xi32>
  %result = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "NE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %result : tensor<4xi1>
}
// CHECK: 4xi8=1 0 1 0

// -----

// CHECK-LABEL: EXEC @compare_lt
func @compare_lt() -> tensor<4xi1> {
  %lhs = constant dense<[1, 2, 7, 4]> : tensor<4xi32>
  %rhs = constant dense<[5, 2, 3, 4]> : tensor<4xi32>
  %result = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "LT"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %result : tensor<4xi1>
}
// CHECK: 4xi8=1 0 0 0

// -----

// CHECK-LABEL: EXEC @compare_le
func @compare_le() -> tensor<4xi1> {
  %lhs = constant dense<[1, 2, 7, 4]> : tensor<4xi32>
  %rhs = constant dense<[5, 2, 3, 4]> : tensor<4xi32>
  %result = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "LE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %result : tensor<4xi1>
}
// CHECK: 4xi8=1 1 0 1

// -----

// CHECK-LABEL: EXEC @compare_gt
func @compare_gt() -> tensor<4xi1> {
  %lhs = constant dense<[1, 2, 7, 4]> : tensor<4xi32>
  %rhs = constant dense<[5, 2, 3, 4]> : tensor<4xi32>
  %result = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "GT"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %result : tensor<4xi1>
}
// CHECK: 4xi8=0 0 1 0

// -----

// CHECK-LABEL: EXEC @compare_ge
func @compare_ge() -> tensor<4xi1> {
  %lhs = constant dense<[1, 2, 7, 4]> : tensor<4xi32>
  %rhs = constant dense<[5, 2, 3, 4]> : tensor<4xi32>
  %result = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "GE"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %result : tensor<4xi1>
}
// CHECK: 4xi8=0 1 1 1
