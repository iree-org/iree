func.func @tensor_float() {
  %0 = util.unfoldable_constant dense<[-1.0, 0.0, 0.5, 1.0]> : tensor<4xf32>
  %result = tosa.tanh %0 : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[-0.761594, 0.0, 0.462117, 0.761594]> : tensor<4xf32>) : tensor<4xf32>
  return
}
