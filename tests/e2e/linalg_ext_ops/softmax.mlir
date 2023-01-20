func.func @softmax() {
  %input = util.unfoldable_constant dense<1.0> : tensor<2x8x4xf32>

  %init = tensor.empty() : tensor<2x8x4xf32>
  %1 = iree_linalg_ext.softmax dimension(2)
       ins(%input : tensor<2x8x4xf32>)
       outs(%init : tensor<2x8x4xf32>) -> tensor<2x8x4xf32>
  check.expect_almost_eq_const(
      %1,
      dense<0.25> : tensor<2x8x4xf32>
  ) : tensor<2x8x4xf32>
  return
}
