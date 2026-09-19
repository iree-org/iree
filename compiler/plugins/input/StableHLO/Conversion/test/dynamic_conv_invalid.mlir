// RUN: iree-opt --iree-stablehlo-input-transformation-pipeline %s --verify-diagnostics

// A dynamic_conv whose result has dynamic spatial dims does not lower yet.
func.func @dynamic_conv_dynamic_spatial(%a: tensor<2x8x6x3xf32>, %k: tensor<3x2x3x4xf32>, %p: tensor<2x2xi64>) -> tensor<2x?x?x4xf32> {
  // expected-error @+1 {{failed to legalize operation 'stablehlo.convolution'}}
  %r = "stablehlo.dynamic_conv"(%a, %k, %p) {
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64, batch_group_count = 1 : i64,
    window_strides = array<i64: 1, 1>, lhs_dilation = array<i64: 1, 1>, rhs_dilation = array<i64: 1, 1>
  } : (tensor<2x8x6x3xf32>, tensor<3x2x3x4xf32>, tensor<2x2xi64>) -> tensor<2x?x?x4xf32>
  return %r : tensor<2x?x?x4xf32>
}
