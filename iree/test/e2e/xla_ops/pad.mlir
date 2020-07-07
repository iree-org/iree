func @pad_test() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %c0 = iree.unfoldable_constant dense<0> : tensor<i32>
  %res = "mhlo.pad"(%input, %c0) {
    edge_padding_low = dense<[0, 1]> : tensor<2xi64>,
    edge_padding_high = dense<[1, 5]> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<2x3xi32>, tensor<i32>) -> tensor<3x9xi32>
  check.expect_eq_const(%res, dense<[
      [0, 1, 2, 3, 0, 0, 0, 0, 0],
      [0, 4, 5, 6, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0]]> : tensor<3x9xi32>) : tensor<3x9xi32>
  return
}

func @pad_no_op() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %c0 = iree.unfoldable_constant dense<0> : tensor<i32>
  %res = "mhlo.pad"(%input, %c0) {edge_padding_high = dense<[0, 0]> : tensor<2xi64>, edge_padding_low = dense<[0, 0]> : tensor<2xi64>, interior_padding = dense<0> : tensor<2xi64>} : (tensor<2x3xi32>, tensor<i32>) -> tensor<2x3xi32>
  check.expect_eq(%res, %input) : tensor<2x3xi32>
  return
}
