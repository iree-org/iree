// Tests the canonicalization/folding of the Check dialect ops.

// RUN: iree-opt --canonicalize --split-input-file %s | FileCheck %s

// CHECK-LABEL: @expect_eq_const
// CHECK-SAME: %[[LHS:[a-zA-Z0-9$._-]+]]
util.func public @expect_eq_const(%lhs : tensor<2x2xi32>) {
  // CHECK: %[[C:.+]] = arith.constant dense<1> : tensor<2x2xi32>
  // CHECK: check.expect_eq(%[[LHS]], %[[C]]) : tensor<2x2xi32>
  check.expect_eq_const(%lhs, dense<1> : tensor<2x2xi32>) : tensor<2x2xi32>
  util.return
}

// -----

// CHECK-LABEL: @expect_almost_eq_const
// CHECK-SAME: %[[LHS:[a-zA-Z0-9$._-]+]]
util.func public @expect_almost_eq_const(%lhs : tensor<2x2xf32>) {
  // CHECK: %[[C:.+]] = arith.constant dense<1.000000e+00> : tensor<2x2xf32>
  // CHECK: check.expect_almost_eq(%[[LHS]], %[[C]]) : tensor<2x2xf32>
  check.expect_almost_eq_const(%lhs, dense<1.0> : tensor<2x2xf32>) : tensor<2x2xf32>
  util.return
}
