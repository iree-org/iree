func @dot_passthrough() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<[[0.3, 0.5]]> : tensor<1x2xf32>
  %rhs = iree.unfoldable_constant  dense<[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]> : tensor<2x3xf32>
  %res = "mhlo.dot"(%lhs, %rhs) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
  check.expect_almost_eq_const(%res, dense<[[0.23, 0.31, 0.39]]> : tensor<1x3xf32>) : tensor<1x3xf32>
  return
}

func @gemm() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<[
    [15.0, 14.0, 13.0],
    [12.0, 11.0, 10.0],
    [09.0, 08.0, 07.0],
    [06.0, 05.0, 04.0],
    [03.0, 02.0, 01.0]]> : tensor<5x3xf32>
  %rhs = iree.unfoldable_constant dense<[
    [15.0, 14.0, 13.0, 12.0, 11.0],
    [10.0, 09.0, 08.0, 07.0, 06.0],
    [05.0, 04.0, 03.0, 02.0, 01.0]]> : tensor<3x5xf32>
  %res = "mhlo.dot"(%lhs, %rhs) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
  check.expect_almost_eq_const(%res, dense<[
    [430.0, 388.0, 346.0, 304.0, 262.0],
    [340.0, 307.0, 274.0, 241.0, 208.0],
    [250.0, 226.0, 202.0, 178.0, 154.0],
    [160.0, 145.0, 130.0, 115.0, 100.0],
    [70.0, 64.0, 58.0, 52.0, 46.0]]> : tensor<5x5xf32>) : tensor<5x5xf32>
  return
}

func @large_matmul() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1.0> : tensor<32x1024xf32>
  %rhs = iree.unfoldable_constant dense<0.4> : tensor<1024x64xf32>
  %res = "mhlo.dot"(%lhs, %rhs) : (tensor<32x1024xf32>, tensor<1024x64xf32>) -> tensor<32x64xf32>
  check.expect_almost_eq_const(%res, dense<409.596> : tensor<32x64xf32>) : tensor<32x64xf32>
  return
}
