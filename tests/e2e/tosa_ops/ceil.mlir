func.func @tensor_float() {
  %0 = util.unfoldable_constant dense<[0.0, -1.3, 1.3, -0.3]> : tensor<4xf32>
  %result = tosa.ceil %0 : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[0.0, -1.0, 2.0, 0.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}
