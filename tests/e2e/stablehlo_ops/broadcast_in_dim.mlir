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

// Runtime operand sizes must select both expanding and nonexpanding indices.
func.func @dynamic_broadcast_expanding() {
  %input = flow.tensor.dynamic_constant dense<[7.0]> : tensor<1xf32> -> tensor<?xf32>
  %shape = util.unfoldable_constant dense<[2, 3]> : tensor<2xi64>
  %result = stablehlo.dynamic_broadcast_in_dim %input, %shape, dims = [1] : (tensor<?xf32>, tensor<2xi64>) -> tensor<2x3xf32>
  check.expect_eq_const(%result, dense<7.0> : tensor<2x3xf32>) : tensor<2x3xf32>
  return
}

func.func @dynamic_broadcast_nonexpanding() {
  %input = flow.tensor.dynamic_constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32> -> tensor<?xf32>
  %shape = util.unfoldable_constant dense<[2, 3]> : tensor<2xi64>
  %result = stablehlo.dynamic_broadcast_in_dim %input, %shape, dims = [1] : (tensor<?xf32>, tensor<2xi64>) -> tensor<2x3xf32>
  check.expect_eq_const(%result, dense<[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]> : tensor<2x3xf32>) : tensor<2x3xf32>
  return
}
