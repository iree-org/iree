func.func @tensor() {
  %0 = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %1 = util.unfoldable_constant dense<2.0> : tensor<3x4xf32>
  %result = "chlo.broadcast_add"(%0, %1) : (tensor<4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  check.expect_almost_eq_const(%result,
    dense<[[3.0, 4.0, 5.0, 6.0],
           [3.0, 4.0, 5.0, 6.0],
           [3.0, 4.0, 5.0, 6.0]]> : tensor<3x4xf32>) : tensor<3x4xf32>
  return
}
