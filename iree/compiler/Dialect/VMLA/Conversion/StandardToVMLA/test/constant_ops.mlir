// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @constant_scalar
func @constant_scalar() -> tensor<i16> attributes { sym_visibility = "private" } {
  // CHECK: = "vmla.constant"() {value = dense<12345> : tensor<i16>}
  %0 = constant dense<12345> : tensor<i16>
  return %0 : tensor<i16>
}

// -----

// CHECK-LABEL: @constant_tensor
func @constant_tensor() -> tensor<4xf32> attributes { sym_visibility = "private" } {
  // CHECK: = "vmla.constant"() {value = dense<[-1.000000e+00, -2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>}
  %0 = constant dense<[-1.0, -2.0, 3.0, 4.0]> : tensor<4xf32>
  return %0 : tensor<4xf32>
}
