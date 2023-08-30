func.func @tensor_float() {
  %0 = util.unfoldable_constant dense<[0.0, 1.0, 0.5, 2.0]> : tensor<4xf32>
  %result = tosa.exp %0 : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[1.0, 2.71828, 1.64872, 7.38906]> : tensor<4xf32>) : tensor<4xf32>
  return
}
