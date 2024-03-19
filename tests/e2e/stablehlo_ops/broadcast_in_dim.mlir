func.func @broadcast_in_dim_2D_3D() {
  %input = util.unfoldable_constant dense<[[1, 2, 3, 4],
                                           [5, 6, 7, 8]]> : tensor<2x4xi32>
  %res = "stablehlo.broadcast_in_dim"(%input) {broadcast_dimensions = array<i64: 1, 2>} : (tensor<2x4xi32>) -> tensor<3x2x4xi32>
  check.expect_eq_const(%res, dense<[
      [[1, 2, 3, 4], [5, 6, 7, 8]],
      [[1, 2, 3, 4], [5, 6, 7, 8]],
      [[1, 2, 3, 4], [5, 6, 7, 8]]]> : tensor<3x2x4xi32>) : tensor<3x2x4xi32>
  return
}

func.func @broadcast_in_dim_3D_scalar() {
  %input = util.unfoldable_constant dense<42> : tensor<i32>
  %res = "stablehlo.broadcast_in_dim"(%input) {broadcast_dimensions = array<i64>} : (tensor<i32>) -> tensor<3x2x4xi32>
  check.expect_eq_const(%res, dense<42> : tensor<3x2x4xi32>) : tensor<3x2x4xi32>
  return
}
