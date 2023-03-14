func.func @tensor_float() {
  %0 = util.unfoldable_constant dense<[-1.0, -0.5, 0.0, 1.0]> : tensor<4xf32>
  %result = "tosa.negate"(%0) : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[1.0, 0.5, 0.0, -1.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func.func @tensor_int() {
  %0 = util.unfoldable_constant dense<[-1, 0, 3, 1]> : tensor<4xi32>
  %result = "tosa.negate"(%0) : (tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[1, 0, -3, -1]> : tensor<4xi32>) : tensor<4xi32>
  return
}
