// RUN: iree-opt --pass-pipeline="builtin.module(func.func(chlo-legalize-to-stablehlo), cse)" \
// RUN:   --split-input-file --verify-diagnostics %s | FileCheck %s

// Check the non-broadcast case for each registered op, then just check a
// representative op for detailed broadcast semantics.

// CHECK-LABEL: @addWithoutBroadcast
func.func @addWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: stablehlo.add %arg0, %arg1
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @dynamicBroadcast
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<?x?xf32>
func.func @dynamicBroadcast(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.add
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----
// CHECK-LABEL: @dynamicBroadcastComplex
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<?x?xf32>
func.func @dynamicBroadcastComplex(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>> {
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.complex
   %0 = chlo.broadcast_complex %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>
  func.return %0 : tensor<?x?xcomplex<f32>>
}

// -----
// CHECK-LABEL: @dynamicBroadcastCompare
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<?x?xf32>
func.func @dynamicBroadcastCompare(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xi1> {
  // CHECK:stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.compare
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = #chlo<comparison_direction EQ>} : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  func.return %0 : tensor<?x?xi1>
}

// -----

// CHECK-LABEL: func @selectv2
func.func @selectv2(%arg0: tensor<2xi1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK: tensor.cast
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.select

  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %0: tensor<2xi32>
}

// CHECK-LABEL: func @selectv2_pred_scalar
func.func @selectv2_pred_scalar(%arg0: tensor<i1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK: stablehlo.select
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %0: tensor<2xi32>
}

// CHECK-LABEL: func @selectv2_broadcast_then
func.func @selectv2_broadcast_then(%arg0: tensor<i1>, %arg1: tensor<8x1xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x8x8xi32> {

  // CHECK:  tensor.cast
  // CHECK:  stablehlo.dynamic_broadcast_in_dim
  // CHECK:  stablehlo.select

  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<8x1xi32>, tensor<2x8x8xi32>) -> tensor<2x8x8xi32>
  func.return %0: tensor<2x8x8xi32>
}

// CHECK-LABEL: func @selectv2_broadcast_else
func.func @selectv2_broadcast_else(%arg0: tensor<i1>, %arg1: tensor<2x8x8xi32>, %arg2: tensor<8x1xi32>) -> tensor<2x8x8xi32> {
  // CHECK: stablehlo.select
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x8x8xi32>, tensor<8x1xi32>) -> tensor<2x8x8xi32>
  func.return %0: tensor<2x8x8xi32>
}

// CHECK-LABEL: func @selectv2_broadcast_pred
func.func @selectv2_broadcast_pred(%arg0: tensor<1xi1>, %arg1: tensor<2x8x8xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x8x8xi32> {
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<1xi1>, tensor<2x8x8xi32>, tensor<2x8x8xi32>) -> tensor<2x8x8xi32>
  func.return %0: tensor<2x8x8xi32>
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.select
}

// CHECK-LABEL: func @selectv2_broadcast_tensor_pred
// CHECK-SAME: (%[[ARG0:.*]]: tensor<3xi1>, %[[ARG1:.*]]: tensor<2x3xf16>, %[[ARG2:.*]]: tensor<2x3xf16>)
func.func @selectv2_broadcast_tensor_pred(%arg0: tensor<3xi1>, %arg1: tensor<2x3xf16>, %arg2: tensor<2x3xf16>) -> tensor<2x3xf16> {
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  func.return %0: tensor<2x3xf16>

  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.select

}

// CHECK-LABEL: func @selectv2_broadcast_all
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x1x1xi1>, %[[ARG1:.*]]: tensor<1x8x1xi32>, %[[ARG2:.*]]: tensor<1x1x8xi32>)
func.func @selectv2_broadcast_all(%arg0: tensor<8x1x1xi1>, %arg1: tensor<1x8x1xi32>, %arg2: tensor<1x1x8xi32>) -> tensor<8x8x8xi32> {
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<8x1x1xi1>, tensor<1x8x1xi32>, tensor<1x1x8xi32>) -> tensor<8x8x8xi32>
  func.return %0: tensor<8x8x8xi32>

  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.select

}

// CHECK-LABEL: func @selectv2_dynamic_ranked
func.func @selectv2_dynamic_ranked(%arg0: tensor<1xi1>, %arg1: tensor<2x?x8xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x?x8xi32> {
  // CHECK: stablehlo.dynamic_broadcast_in_dim
  // CHECK: stablehlo.select

  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<1xi1>, tensor<2x?x8xi32>, tensor<2x8x8xi32>) -> tensor<2x?x8xi32>
  func.return %0: tensor<2x?x8xi32>
}

// -----
// Verifies that broadcast_dimensions validity checks are valid.
// CHECK-LABEL: @dynamicNonScalarBroadcastDimensions
func.func @dynamicNonScalarBroadcastDimensions(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK: stablehlo.add
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = array<i64: 1>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  func.return %0 : tensor<1x4xf32>
}

// -----
// Verifies that broadcast_dimensions validity checks are valid.
// CHECK-LABEL: @dynamicNonScalarByScalarBroadcastDimensions
func.func @dynamicNonScalarByScalarBroadcastDimensions(%arg0: tensor<1x4xf32>, %arg1: tensor<f32>) -> tensor<1x4xf32> {
  // CHECK: stablehlo.add
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = array<i64>} : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
  func.return %0 : tensor<1x4xf32>
}

// -----
// Verifies that invalid broadcast dimensions are rejected.
func.func @dynamicNonScalarBroadcastDimensionsSizeMismatch(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // expected-warning @+2 {{unsupported non prefix-padded dynamic rank broadcast_dimensions}}
  // expected-error @+1 {{failed to legalize operation}}
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = array<i64: 1, 2>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  func.return %0 : tensor<1x4xf32>
}

// -----
// Verifies that invalid broadcast dimensions are rejected.
func.func @dynamicNonScalarBroadcastDimensionsMismatch(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // expected-warning @+2 {{unsupported non prefix-padded dynamic rank broadcast_dimensions}}
  // expected-error @+1 {{failed to legalize operation}}
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = array<i64: 2>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  func.return %0 : tensor<1x4xf32>
}

// -----
// Note that broadcast_add is used as a proxy for all of the template
// expansions. Tests below merely verify that the op has an expansion.
// CHECK-LABEL: @andWithoutBroadcast
func.func @andWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK: stablehlo.and %arg0, %arg1
  %0 = chlo.broadcast_and %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @atan2WithoutBroadcast
func.func @atan2WithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: stablehlo.atan2 %arg0, %arg1
  %0 = chlo.broadcast_atan2 %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @compareWithoutBroadcast
func.func @compareWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  // CHECK: stablehlo.compare EQ, %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = #chlo<comparison_direction EQ>} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @complexWithoutBroadcast
func.func @complexWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xcomplex<f32>> {
  // CHECK: stablehlo.complex %arg0, %arg1 : tensor<4xcomplex<f32>>
  %0 = chlo.broadcast_complex %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  func.return %0 : tensor<4xcomplex<f32>>
}

// -----
// CHECK-LABEL: @divideWithoutBroadcast
func.func @divideWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: stablehlo.divide %arg0, %arg1
  %0 = chlo.broadcast_divide %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @maximumWithoutBroadcast
func.func @maximumWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: stablehlo.maximum %arg0, %arg1
  %0 = chlo.broadcast_maximum %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @minimumWithoutBroadcast
func.func @minimumWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: stablehlo.minimum %arg0, %arg1
  %0 = chlo.broadcast_minimum %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @multiplyWithoutBroadcast
func.func @multiplyWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: stablehlo.multiply %arg0, %arg1
  %0 = chlo.broadcast_multiply %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @orWithoutBroadcast
func.func @orWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK: stablehlo.or %arg0, %arg1
  %0 = chlo.broadcast_or %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @powerWithoutBroadcast
func.func @powerWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: stablehlo.power %arg0, %arg1
  %0 = chlo.broadcast_power %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @remainderWithoutBroadcast
func.func @remainderWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: stablehlo.remainder %arg0, %arg1
  %0 = chlo.broadcast_remainder %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @shift_leftWithoutBroadcast
func.func @shift_leftWithoutBroadcast(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: stablehlo.shift_left %arg0, %arg1
  %0 = chlo.broadcast_shift_left %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: @shift_right_arithmeticWithoutBroadcast
func.func @shift_right_arithmeticWithoutBroadcast(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: stablehlo.shift_right_arithmetic %arg0, %arg1
  %0 = chlo.broadcast_shift_right_arithmetic %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: @shift_right_logicalWithoutBroadcast
func.func @shift_right_logicalWithoutBroadcast(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: stablehlo.shift_right_logical %arg0, %arg1
  %0 = chlo.broadcast_shift_right_logical %arg0, %arg1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: @subWithoutBroadcast
func.func @subWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: stablehlo.subtract %arg0, %arg1
  %0 = chlo.broadcast_subtract %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @xorWithoutBroadcast
func.func @xorWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK: stablehlo.xor %arg0, %arg1
  %0 = chlo.broadcast_xor %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}
