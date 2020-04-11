func @dot_passthrough() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<[[0.3, 0.5]]> : tensor<1x2xf32>
  %rhs = iree.unfoldable_constant  dense<[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]> : tensor<2x3xf32>
  %res = "xla_hlo.dot"(%lhs, %rhs) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
  check.expect_almost_eq_const(%res, dense<[[0.23, 0.31, 0.39]]> : tensor<1x3xf32>) : tensor<1x3xf32>
  return
}
