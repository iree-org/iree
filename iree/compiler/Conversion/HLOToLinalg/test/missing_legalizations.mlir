// RUN: iree-opt -split-input-file -iree-codegen-hlo-to-linalg-on-tensors='direct-hlo-client-lowering' -verify-diagnostics %s

// Left and right shift operators are missing HLO->Linalg lowerings.
func @shift_leftWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{failed to legalize operation 'chlo.broadcast_shift_left' that was explicitly marked illegal}}
  %0 = chlo.broadcast_shift_left %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
func @shift_leftWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{failed to legalize operation 'chlo.broadcast_shift_left' that was explicitly marked illegal}}
  %0 = chlo.broadcast_shift_left %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
func @shift_right_arithmeticWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{failed to legalize operation 'chlo.broadcast_shift_right_arithmetic' that was explicitly marked illegal}}
  %0 = chlo.broadcast_shift_right_arithmetic %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
func @shift_right_logicalWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{failed to legalize operation 'chlo.broadcast_shift_right_logical' that was explicitly marked illegal}}
  %0 = chlo.broadcast_shift_right_logical %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// In Feb 2021, the CHLO select op was changed, without update to classic XLA
// or the specification, to follow TF semantics, which allows for a more
// complicated ternary broadcast. This is not yet supported.
// -----
func @selectv2_broadcast_then(%arg0: tensor<i1>, %arg1: tensor<8x1xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x8x8xi32> {
  // expected-error@+1 {{failed to legalize operation 'chlo.broadcast_select' that was explicitly marked illegal}}
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<8x1xi32>, tensor<2x8x8xi32>) -> tensor<2x8x8xi32>
  return %0: tensor<2x8x8xi32>
}

// -----
func @selectv2_broadcast_else(%arg0: tensor<i1>, %arg1: tensor<2x8x8xi32>, %arg2: tensor<8x1xi32>) -> tensor<2x8x8xi32> {
  // expected-error@+1 {{failed to legalize operation 'chlo.broadcast_select' that was explicitly marked illegal}}
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x8x8xi32>, tensor<8x1xi32>) -> tensor<2x8x8xi32>
  return %0: tensor<2x8x8xi32>
}

// -----
func @selectv2_broadcast_pred(%arg0: tensor<1xi1>, %arg1: tensor<2x8x8xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x8x8xi32> {
  // expected-error@+1 {{failed to legalize operation 'chlo.broadcast_select' that was explicitly marked illegal}}
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<1xi1>, tensor<2x8x8xi32>, tensor<2x8x8xi32>) -> tensor<2x8x8xi32>
  return %0: tensor<2x8x8xi32>
}

// -----
func @selectv2_broadcast_tensor_pred(%arg0: tensor<3xi1>, %arg1: tensor<2x3xf16>, %arg2: tensor<2x3xf16>) -> tensor<2x3xf16> {
  // expected-error@+1 {{failed to legalize operation 'chlo.broadcast_select' that was explicitly marked illegal}}
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x3xf16>, tensor<2x3xf16>) -> tensor<2x3xf16>
  return %0: tensor<2x3xf16>
}

// -----
func @selectv2_broadcast_all(%arg0: tensor<8x1x1xi1>, %arg1: tensor<1x8x1xi32>, %arg2: tensor<1x1x8xi32>) -> tensor<8x8x8xi32> {
  // expected-error@+1 {{failed to legalize operation 'chlo.broadcast_select' that was explicitly marked illegal}}
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<8x1x1xi1>, tensor<1x8x1xi32>, tensor<1x1x8xi32>) -> tensor<8x8x8xi32>
  return %0: tensor<8x8x8xi32>
}

// -----
func @selectv2_dynamic_ranked(%arg0: tensor<1xi1>, %arg1: tensor<2x?x8xi32>, %arg2: tensor<2x8x8xi32>) -> tensor<2x?x8xi32> {
  // expected-error@+1 {{failed to legalize operation 'chlo.broadcast_select' that was explicitly marked illegal}}
  %0 = "chlo.broadcast_select"(%arg0, %arg1, %arg2) : (tensor<1xi1>, tensor<2x?x8xi32>, tensor<2x8x8xi32>) -> tensor<2x?x8xi32>
  return %0: tensor<2x?x8xi32>
}

// -----
// Non-numpy compatible broadcast_dimensions are not supported.
// Note: This is by design and support is not planned.
func @dynamicNonScalarBroadcastDimensionsSizeMismatch(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // expected-error@+1 {{failed to legalize operation 'chlo.broadcast_add' that was explicitly marked illegal}}
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// Non-numpy compatible broadcast_dimensions are not supported.
// Note: This is by design and support is not planned.
func @dynamicNonScalarBroadcastDimensionsMismatch(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // expected-error@+1 {{failed to legalize operation 'chlo.broadcast_add' that was explicitly marked illegal}}
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
