func @i32() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[5, 6, 7, 8]> : tensor<4xi32>
  %1 = iree.unfoldable_constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %result = "mhlo.divide"(%0, %1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[5, 3, 2, 2]> : tensor<4xi32>) : tensor<4xi32>
  return
}

func @f32() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>
  %1 = iree.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %result = "mhlo.divide"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[5.0, 3.0, 2.333333, 2.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}
