func.func @tensor_float() {
  %0 = util.unfoldable_constant dense<[1.0, 5.0, 0.5, 2.0]> : tensor<4xf32>
  %result = tosa.log %0 : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[0.0, 1.60943, -0.693147, 0.693147]> : tensor<4xf32>) : tensor<4xf32>
  return
}
