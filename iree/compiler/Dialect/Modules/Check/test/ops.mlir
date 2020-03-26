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

// Tests the printing/parsing of the Check dialect ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @expect_true
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @expect_true(%arg : i32) {
  // CHECK: check.expect_true([[ARG]]) : i32
  check.expect_true(%arg) : i32
  return
}

// -----

// CHECK-LABEL: @expect_false
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @expect_false(%arg : i32) {
  // CHECK: check.expect_false([[ARG]]) : i32
  check.expect_false(%arg) : i32
  return
}

// -----

// CHECK-LABEL: @expect_all_true
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @expect_all_true(%arg : !hal.buffer_view) {
  // CHECK: check.expect_all_true([[ARG]]) : !hal.buffer_view
  check.expect_all_true(%arg) : !hal.buffer_view
  return
}

// -----

// CHECK-LABEL: @expect_all_true_tensor
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @expect_all_true_tensor(%arg : tensor<2x2xi32>) {
  // CHECK: check.expect_all_true([[ARG]]) : tensor<2x2xi32>
  check.expect_all_true(%arg) : tensor<2x2xi32>
  return
}

// -----

// CHECK-LABEL: @expect_eq
// CHECK-SAME: [[LHS:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[RHS:%[a-zA-Z0-9]+]]
func @expect_eq(%lhs : !hal.buffer_view, %rhs : !hal.buffer_view) {
  // CHECK: check.expect_eq([[LHS]], [[RHS]]) : !hal.buffer_view
  check.expect_eq(%lhs, %rhs) : !hal.buffer_view
  return
}

// -----

// CHECK-LABEL: @expect_eq_tensor
// CHECK-SAME: [[LHS:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[RHS:%[a-zA-Z0-9]+]]
func @expect_eq_tensor(%lhs : tensor<2x2xi32>, %rhs : tensor<2x2xi32>) {
  // CHECK: check.expect_eq([[LHS]], [[RHS]]) : tensor<2x2xi32>
  check.expect_eq(%lhs, %rhs) : tensor<2x2xi32>
  return
}

// -----

// CHECK-LABEL: @expect_almost_eq
// CHECK-SAME: [[LHS:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[RHS:%[a-zA-Z0-9]+]]
func @expect_almost_eq(%lhs : !hal.buffer_view, %rhs : !hal.buffer_view) {
  // CHECK: check.expect_almost_eq([[LHS]], [[RHS]]) : !hal.buffer_view
  check.expect_almost_eq(%lhs, %rhs) : !hal.buffer_view
  return
}

// -----

// CHECK-LABEL: @expect_almost_eq_tensor
// CHECK-SAME: [[LHS:%[a-zA-Z0-9]+]]
// CHECK-SAME: [[RHS:%[a-zA-Z0-9]+]]
func @expect_almost_eq_tensor(%lhs : tensor<2x2xf32>, %rhs : tensor<2x2xf32>) {
  // CHECK: check.expect_almost_eq([[LHS]], [[RHS]]) : tensor<2x2xf32>
  check.expect_almost_eq(%lhs, %rhs) : tensor<2x2xf32>
  return
}
