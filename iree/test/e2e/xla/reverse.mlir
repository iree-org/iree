func @xla_reverse() attributes { iree.module.export } {
  %t1 = iree.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>

  %dim0 = "xla_hlo.reverse"(%t1) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %expected_dim0 = iree.unfoldable_constant dense<[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]> : tensor<2x3xf32>
  check.expect_almost_eq(%dim0, %expected_dim0) : tensor<2x3xf32>

  %dim1 = "xla_hlo.reverse"(%t1) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %expected_dim1 = iree.unfoldable_constant dense<[[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]> : tensor<2x3xf32>
  check.expect_almost_eq(%dim1, %expected_dim1) : tensor<2x3xf32>

  %both_dims = "xla_hlo.reverse"(%t1) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %expected_both_dims = iree.unfoldable_constant dense<[[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]> : tensor<2x3xf32>
  check.expect_almost_eq(%both_dims, %expected_both_dims) : tensor<2x3xf32>
  return
}
