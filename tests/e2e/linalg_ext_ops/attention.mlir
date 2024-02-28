func.func @attention() {
  %init = tensor.empty() : tensor<1x4x4xf32>
  %query = util.unfoldable_constant dense<1.0> : tensor<1x4x4xf32>
  %key = util.unfoldable_constant dense<0.5> : tensor<1x4x4xf32>
  %value = util.unfoldable_constant dense<2.0> : tensor<1x4x4xf32>
  %1 = iree_linalg_ext.attention ins(%query, %key, %value : tensor<1x4x4xf32>,
        tensor<1x4x4xf32>, tensor<1x4x4xf32>) outs(%init : tensor<1x4x4xf32>) -> tensor<1x4x4xf32>
  check.expect_almost_eq_const(
      %1,
      dense<[[[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]]]> : tensor<1x4x4xf32>
  ) : tensor<1x4x4xf32>
  return
}
