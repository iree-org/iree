func @reshape_1D_2D() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]> : tensor<12xi32>
  %result = "xla_hlo.reshape"(%input) : (tensor<12xi32>) -> tensor<3x4xi32>
  check.expect_eq_const(%result, dense<[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12]]> : tensor<3x4xi32>) : tensor<3x4xi32>
  return
}

func @reshape_1D_3D() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]> : tensor<12xi32>
  %result = "xla_hlo.reshape"(%input) : (tensor<12xi32>) -> tensor<2x2x3xi32>
  check.expect_eq_const(%result, dense<[
      [[1, 2, 3], [4, 5, 6]],
      [[7, 8, 9], [10, 11, 12]]]> : tensor<2x2x3xi32>) : tensor<2x2x3xi32>
  return
}
