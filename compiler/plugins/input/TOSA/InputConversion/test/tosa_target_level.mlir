// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline="builtin.module(iree-tosa-input-transformation-pipeline)" \
// RUN:   --verify-diagnostics %s | FileCheck %s

// IREE targets TOSA level=none so that models with dynamic dimensions
// (e.g. a dynamic batch size) compile, but we're still able to run profile and
// data type validation.

// CHECK-LABEL: @dynamic_shape
func.func @dynamic_shape(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
  // CHECK: linalg.generic
  %0 = tosa.clamp %arg0 {max_val = 1.0 : f32, min_val = 0.0 : f32} : (tensor<?x4xf32>) -> tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

// -----

// Genuine profile/data type violations, unrelated to shape, must still be
// rejected.

// CHECK-LABEL: @out_of_profile
func.func @out_of_profile(%arg0: tensor<4xf8E4M3FN>) -> tensor<4xf8E4M3FN> {
  // expected-error@+1 {{operand/result data types did not align with any profile or extension}}
  %0 = tosa.abs %arg0 : (tensor<4xf8E4M3FN>) -> tensor<4xf8E4M3FN>
  return %0 : tensor<4xf8E4M3FN>
}
