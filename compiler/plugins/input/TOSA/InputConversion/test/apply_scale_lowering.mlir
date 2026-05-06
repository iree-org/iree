// RUN: iree-compile --compile-to=input --split-input-file %s | FileCheck %s

// Make sure that tosa.apply_scale operations generated during TOSA to
// linalg lowering, are properly lowered during TOSA to arith lowering

// tosa.mul with a non-zero integer shift lowers through tosa.apply_scale
//
// CHECK-LABEL: util.func public @shifted_mul
// CHECK-NOT: tosa.apply_scale
// CHECK: return
func.func @shifted_mul(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %shift = "tosa.const"() {values = dense<1> : tensor<1xi8>} : () -> tensor<1xi8>
  %0 = tosa.mul %arg0, %arg1, %shift : (tensor<4xi32>, tensor<4xi32>, tensor<1xi8>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// Quantized tosa.avg_pool2d also lowers through tosa.apply_scale
//
// CHECK-LABEL: util.func public @quantized_avg_pool
// CHECK-NOT: tosa.apply_scale
// CHECK: return
func.func @quantized_avg_pool(%arg0: tensor<1x2x4x1xi8>) -> tensor<1x1x3x1xi8> {
  %input_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %0 = tosa.avg_pool2d %arg0, %input_zp, %output_zp {acc_type = i32, kernel = array<i64: 2, 2>, stride = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>} : (tensor<1x2x4x1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x3x1xi8>
  return %0 : tensor<1x1x3x1xi8>
}
