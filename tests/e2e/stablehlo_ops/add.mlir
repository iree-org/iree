func.func @tensor() {
  %0 = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %1 = util.unfoldable_constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>
  %result = "stablehlo.add"(%0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[6.0, 8.0, 10.0, 12.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func.func @tensor_4d() {
  %0 = util.unfoldable_constant dense<[[[[1.0, 2.0], [3.0, 4.0]],
                                         [[5.0, 6.0], [7.0, 8.0]]],
                                        [[[9.0, 10.0], [11.0, 12.0]],
                                         [[13.0, 14.0], [15.0, 16.0]]]]> :
    tensor<2x2x2x2xf32>
  %1 = util.unfoldable_constant dense<[[[[1.0, 2.0], [3.0, 4.0]],
                                         [[5.0, 6.0], [7.0, 8.0]]],
                                        [[[9.0, 10.0], [11.0, 12.0]],
                                         [[13.0, 14.0], [15.0, 16.0]]]]> :
    tensor<2x2x2x2xf32>
  %result = "stablehlo.add"(%0, %1) : (tensor<2x2x2x2xf32>, tensor<2x2x2x2xf32>)
    -> tensor<2x2x2x2xf32>
  check.expect_almost_eq_const(%result, dense<[[[[2.0, 4.0], [6.0, 8.0]],
                                               [[10.0, 12.0], [14.0, 16.0]]],
                                              [[[18.0, 20.0], [22.0, 24.0]],
                                               [[26.0, 28.0], [30.0, 32.0]]]]> :
    tensor<2x2x2x2xf32>) : tensor<2x2x2x2xf32>
  return
}
