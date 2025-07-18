// RUN: iree-compile --iree-hal-target-device=local --iree-hal-local-target-device-backends=vmvx --iree-input-demote-f64-to-f32=false %s | iree-check-module --module=- --expect_failure | FileCheck %s

// CHECK-LABEL: expect_true_of_false
// CHECK: Expected 0 to be nonzero
func.func @expect_true_of_false() {
  %false = util.unfoldable_constant 0 : i32
  check.expect_true(%false) : i32
  return
}

// CHECK-LABEL: expect_almost_eq_const_f32
// CHECK: Expected near equality of these values. Contents does not match to tolerance parameters atol=0.0001, rtol=0. The first failure occurs at index 0 as the lhs value 1 differs from the rhs value 0.999 by -1.000e-03.
func.func @expect_almost_eq_const_f32() {
  %const0 = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf32>
  check.expect_almost_eq_const(%const0, dense<[0.999, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf32>) : tensor<5xf32>
  return
}

// CHECK-LABEL: expect_almost_eq_const_f64
// CHECK: Expected near equality of these values. Contents does not match to tolerance parameters atol=0.01, rtol=0. The first failure occurs at index 0 as the lhs value 1 differs from the rhs value 0.98 by -2.000e-02.
func.func @expect_almost_eq_const_f64() {
  %const0 = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf64>
  check.expect_almost_eq_const(%const0, dense<[0.98, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf64>, atol 1.0e-2) : tensor<5xf64>
  return
}

// CHECK-LABEL: expect_almost_eq_const_f16
// CHECK: Expected near equality of these values. Contents does not match to tolerance parameters atol=0.01, rtol=0.01. The first failure occurs at index 0 as the lhs value 0 differs from the rhs value 0.0110016 by 1.100e-02.
func.func @expect_almost_eq_const_f16() {
  %const0 = util.unfoldable_constant dense<[0.0, 100.0]> : tensor<2xf16>
  check.expect_almost_eq_const(%const0, dense<[0.011, 99.0]> : tensor<2xf16>, atol 1.0e-2, rtol 1.0e-2) : tensor<2xf16>
  return
}

// CHECK-LABEL: expect_almost_eq_const_bf16
// CHECK: Expected near equality of these values. Contents does not match to tolerance parameters atol=0.01, rtol=0.01. The first failure occurs at index 1 as the lhs value 100 differs from the rhs value 98 by -2.000e+00.
func.func @expect_almost_eq_const_bf16() {
  %const0 = util.unfoldable_constant dense<[0.0, 100.0]> : tensor<2xbf16>
  check.expect_almost_eq_const(%const0, dense<[0.009, 98.0]> : tensor<2xbf16>, atol 1.0e-2, rtol 1.0e-2) : tensor<2xbf16>
  return
}

// CHECK-LABEL: expect_almost_eq_const_f8E5M2
// CHECK: Expected near equality of these values. Contents does not match to tolerance parameters atol=0.1, rtol=0.1. The first failure occurs at index 0 as the lhs value 0 differs from the rhs value 0.25 by 2.500e-01.
func.func @expect_almost_eq_const_f8E5M2() {
  %const0 = util.unfoldable_constant dense<[0.0, 100.0]> : tensor<2xf8E5M2>
  check.expect_almost_eq_const(%const0, dense<[0.25, 75.0]> : tensor<2xf8E5M2>, atol 0.1, rtol 0.1) : tensor<2xf8E5M2>
  return
}

// CHECK: Test failed as expected
