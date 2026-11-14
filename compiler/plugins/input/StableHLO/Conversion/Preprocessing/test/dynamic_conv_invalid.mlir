// RUN: iree-opt --iree-stablehlo-to-stablehlo-preprocessing --verify-diagnostics --split-input-file %s

// -----

func.func @invalid_unsigned_padding_low(%a: tensor<1x2x1xf32>, %k: tensor<1x1x1xf32>) -> tensor<1x?x1xf32> {
  %p = stablehlo.constant dense<[[18446744073709551615, 0]]> : tensor<1x2xui64>
  // expected-error@+1 {{padding exceeds the signed 64-bit dimension range}}
  %r = "stablehlo.dynamic_conv"(%a, %k, %p) {
    dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
    feature_group_count = 1 : i64, batch_group_count = 1 : i64,
    window_strides = array<i64: 1>, lhs_dilation = array<i64: 1>, rhs_dilation = array<i64: 1>
  } : (tensor<1x2x1xf32>, tensor<1x1x1xf32>, tensor<1x2xui64>) -> tensor<1x?x1xf32>
  return %r : tensor<1x?x1xf32>
}

// -----

func.func @invalid_unsigned_padding_high(%a: tensor<1x2x1xf32>, %k: tensor<1x1x1xf32>) -> tensor<1x?x1xf32> {
  %p = stablehlo.constant dense<[[0, 18446744073709551615]]> : tensor<1x2xui64>
  // expected-error@+1 {{padding exceeds the signed 64-bit dimension range}}
  %r = "stablehlo.dynamic_conv"(%a, %k, %p) {
    dimension_numbers = #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>,
    feature_group_count = 1 : i64, batch_group_count = 1 : i64,
    window_strides = array<i64: 1>, lhs_dilation = array<i64: 1>, rhs_dilation = array<i64: 1>
  } : (tensor<1x2x1xf32>, tensor<1x1x1xf32>, tensor<1x2xui64>) -> tensor<1x?x1xf32>
  return %r : tensor<1x?x1xf32>
}
