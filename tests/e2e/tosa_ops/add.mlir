func.func @tensor_float() {
  %0 = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %1 = util.unfoldable_constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>
  %result = tosa.add %0, %1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[6.0, 8.0, 10.0, 12.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func.func @tensor_int() {
  %0 = util.unfoldable_constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %1 = util.unfoldable_constant dense<[5, 6, 7, 8]> : tensor<4xi32>
  %result = tosa.add %0, %1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[6, 8, 10, 12]> : tensor<4xi32>) : tensor<4xi32>
  return
}
