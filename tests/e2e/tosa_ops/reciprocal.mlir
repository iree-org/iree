func.func @tensor_float() {
  %0 = util.unfoldable_constant dense<[-1.0, -0.5, 10.0, 2.0]> : tensor<4xf32>
  %result = tosa.reciprocal %0 : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[-1.0, -2.0, 0.1, 0.5]> : tensor<4xf32>) : tensor<4xf32>
  return
}
