func.func @tensor_float() {
  %0 = util.unfoldable_constant dense<[0.0, 1.0, 50.0, 100.0]> : tensor<4xf32>
  %result = tosa.sigmoid %0 : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[0.5, 0.7310586, 1.0, 1.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}
