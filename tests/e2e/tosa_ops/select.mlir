func.func @tensor_float() {
  %0 = util.unfoldable_constant dense<[0, 0, 1, 1]> : tensor<4xi1>
  %1 = util.unfoldable_constant dense<[1.0, 5.0, 3.0, 4.0]> : tensor<4xf32>
  %2 = util.unfoldable_constant dense<[5.0, 1.0, 3.0, 1.5]> : tensor<4xf32>
  %result = tosa.select %0, %1, %2 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[5.0, 1.0, 3.0, 4.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func.func @tensor_int() {
  %0 = util.unfoldable_constant dense<[0, 0, 1, 1]> : tensor<4xi1>
  %1 = util.unfoldable_constant dense<[1, 5, 3, 4]> : tensor<4xi32>
  %2 = util.unfoldable_constant dense<[5, 1, 3, 1]> : tensor<4xi32>
  %result = tosa.select %0, %1, %2 : (tensor<4xi1>, tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[5, 1, 3, 4]> : tensor<4xi32>) : tensor<4xi32>
  return
}
