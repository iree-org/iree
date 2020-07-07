func @xla_reverse() attributes { iree.module.export } {
  %t1 = iree.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>

  %dim0 = "mhlo.reverse"(%t1) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  check.expect_almost_eq_const(
      %dim0,
      dense<[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]> : tensor<2x3xf32>
  ) : tensor<2x3xf32>

  %dim1 = "mhlo.reverse"(%t1) {dimensions = dense<1> : tensor<1xi64>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  check.expect_almost_eq_const(
      %dim1,
      dense<[[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]> : tensor<2x3xf32>
  ) : tensor<2x3xf32>

  %both_dims = "mhlo.reverse"(%t1) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  check.expect_almost_eq_const(
      %both_dims,
      dense<[[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]> : tensor<2x3xf32>
  ) : tensor<2x3xf32>
  return
}
