func @tensor() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %1 = iree.unfoldable_constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>
  %result = "mhlo.add"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[6.0, 8.0, 10.0, 12.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}
