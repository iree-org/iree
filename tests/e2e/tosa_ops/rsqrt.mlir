func.func @tensor_float() {
  %0 = util.unfoldable_constant dense<[16.0, 4.0, 9.0, 1.0]> : tensor<4xf32>
  %result = tosa.rsqrt %0 : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[0.25, 0.5, 0.3333, 1.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}
