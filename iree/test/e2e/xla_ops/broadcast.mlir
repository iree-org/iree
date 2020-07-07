func @broadcast_2D_3D() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[[1, 2, 3, 4],
                                           [5, 6, 7, 8]]> : tensor<2x4xi32>
  %result = "mhlo.broadcast"(%input) {broadcast_sizes = dense<3> : tensor<1xi64>} : (tensor<2x4xi32>) -> tensor<3x2x4xi32>
  check.expect_eq_const(%result, dense<[
      [[1, 2, 3, 4], [5, 6, 7, 8]],
      [[1, 2, 3, 4], [5, 6, 7, 8]],
      [[1, 2, 3, 4], [5, 6, 7, 8]]]> : tensor<3x2x4xi32>) : tensor<3x2x4xi32>
  return
}

func @broadcast_3D_scalar() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<42> : tensor<i32>
  %result = "mhlo.broadcast"(%input) {broadcast_sizes = dense<[3, 2, 4]> : tensor<3xi64>} : (tensor<i32>) -> tensor<3x2x4xi32>
  check.expect_eq_const(%result, dense<[
      [[42, 42, 42, 42], [42, 42, 42, 42]],
      [[42, 42, 42, 42], [42, 42, 42, 42]],
      [[42, 42, 42, 42], [42, 42, 42, 42]]]> : tensor<3x2x4xi32>) : tensor<3x2x4xi32>
  return
}
