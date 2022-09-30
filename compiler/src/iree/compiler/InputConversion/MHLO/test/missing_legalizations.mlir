// RUN: iree-opt --split-input-file --iree-mhlo-to-linalg-on-tensors --verify-diagnostics %s

// Non-numpy compatible broadcast_dimensions are not supported.
// Note: This is by design and support is not planned.
func.func @dynamicNonScalarBroadcastDimensionsSizeMismatch(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // expected-error@+1 {{failed to legalize operation 'chlo.broadcast_add' that was explicitly marked illegal}}
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// Non-numpy compatible broadcast_dimensions are not supported.
// Note: This is by design and support is not planned.
func.func @dynamicNonScalarBroadcastDimensionsMismatch(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // expected-error@+1 {{failed to legalize operation 'chlo.broadcast_add' that was explicitly marked illegal}}
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
