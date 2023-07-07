func.func @i32() {
  %0 = util.unfoldable_constant dense<[5, 6, 3, 4]> : tensor<4xi32>
  %1 = util.unfoldable_constant dense<[1, 4, 7, 6]> : tensor<4xi32>
  %result = "stablehlo.subtract"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[4, 2, -4, -2]> : tensor<4xi32>) : tensor<4xi32>
  return
}

func.func @f32() {
  %0 = util.unfoldable_constant dense<[5.0, 6.0, 3.0, 4.0]> : tensor<4xf32>
  %1 = util.unfoldable_constant dense<[1.0, 4.0, 7.0, 6.0]> : tensor<4xf32>
  %result = "stablehlo.subtract"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[4.0, 2.0, -4.0, -2.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}
