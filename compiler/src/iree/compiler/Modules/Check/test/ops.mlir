// Tests the printing/parsing of the Check dialect ops.

// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @expect_true
// CHECK-SAME: %[[ARG:[a-zA-Z0-9$._-]+]]
util.func public @expect_true(%arg : i32) {
  // CHECK: check.expect_true(%[[ARG]]) : i32
  check.expect_true(%arg) : i32
  util.return
}

// -----

// CHECK-LABEL: @expect_false
// CHECK-SAME: %[[ARG:[a-zA-Z0-9$._-]+]]
util.func public @expect_false(%arg : i32) {
  // CHECK: check.expect_false(%[[ARG]]) : i32
  check.expect_false(%arg) : i32
  util.return
}

// -----

// CHECK-LABEL: @expect_all_true
// CHECK-SAME: %[[ARG:[a-zA-Z0-9$._-]+]]
util.func public @expect_all_true(%arg : !hal.buffer_view) {
  // CHECK: check.expect_all_true(%[[ARG]]) : !hal.buffer_view
  check.expect_all_true(%arg) : !hal.buffer_view
  util.return
}

// -----

// CHECK-LABEL: @expect_all_true_tensor
// CHECK-SAME: %[[ARG:[a-zA-Z0-9$._-]+]]
util.func public @expect_all_true_tensor(%arg : tensor<2x2xi32>) {
  // CHECK: check.expect_all_true(%[[ARG]]) : tensor<2x2xi32>
  check.expect_all_true(%arg) : tensor<2x2xi32>
  util.return
}

// -----

// CHECK-LABEL: @expect_eq
// CHECK-SAME: %[[LHS:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[RHS:[a-zA-Z0-9$._-]+]]
util.func public @expect_eq(%lhs : !hal.buffer_view, %rhs : !hal.buffer_view) {
  // CHECK: check.expect_eq(%[[LHS]], %[[RHS]]) : !hal.buffer_view
  check.expect_eq(%lhs, %rhs) : !hal.buffer_view
  util.return
}

// -----

// CHECK-LABEL: @expect_eq_tensor
// CHECK-SAME: %[[LHS:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[RHS:[a-zA-Z0-9$._-]+]]
util.func public @expect_eq_tensor(%lhs : tensor<2x2xi32>, %rhs : tensor<2x2xi32>) {
  // CHECK: check.expect_eq(%[[LHS]], %[[RHS]]) : tensor<2x2xi32>
  check.expect_eq(%lhs, %rhs) : tensor<2x2xi32>
  util.return
}

// -----

// CHECK-LABEL: @expect_eq_const
// CHECK-SAME: %[[LHS:[a-zA-Z0-9$._-]+]]
util.func public @expect_eq_const(%lhs : tensor<2x2xi32>) {
  // CHECK: check.expect_eq_const(%[[LHS]], dense<1> : tensor<2x2xi32>) : tensor<2x2xi32>
  check.expect_eq_const(%lhs, dense<1> : tensor<2x2xi32>) : tensor<2x2xi32>
  util.return
}

// -----

// CHECK-LABEL: @expect_almost_eq
// CHECK-SAME: %[[LHS:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[RHS:[a-zA-Z0-9$._-]+]]
util.func public @expect_almost_eq(%lhs : !hal.buffer_view, %rhs : !hal.buffer_view) {
  // CHECK: check.expect_almost_eq(%[[LHS]], %[[RHS]]) : !hal.buffer_view
  check.expect_almost_eq(%lhs, %rhs) : !hal.buffer_view
  util.return
}

// -----

// CHECK-LABEL: @expect_almost_eq_tensor
// CHECK-SAME: %[[LHS:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[RHS:[a-zA-Z0-9$._-]+]]
util.func public @expect_almost_eq_tensor(%lhs : tensor<2x2xf32>, %rhs : tensor<2x2xf32>) {
  // CHECK: check.expect_almost_eq(%[[LHS]], %[[RHS]]) : tensor<2x2xf32>
  check.expect_almost_eq(%lhs, %rhs) : tensor<2x2xf32>
  util.return
}

// -----

// CHECK-LABEL: @expect_almost_eq_const
// CHECK-SAME: %[[LHS:[a-zA-Z0-9$._-]+]]
util.func public @expect_almost_eq_const(%lhs : tensor<2x2xf32>) {
  // CHECK: check.expect_almost_eq_const(%[[LHS]], dense<1.000000e+00> : tensor<2x2xf32>) : tensor<2x2xf32>
  check.expect_almost_eq_const(%lhs, dense<1.0> : tensor<2x2xf32>) : tensor<2x2xf32>
  util.return
}
