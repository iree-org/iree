func.func @xla_reverse() {
  %t1 = util.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>

  %dim0 = "stablehlo.reverse"(%t1) {dimensions = array<i64: 0>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  check.expect_almost_eq_const(
      %dim0,
      dense<[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]> : tensor<2x3xf32>
  ) : tensor<2x3xf32>

  %dim1 = "stablehlo.reverse"(%t1) {dimensions = array<i64: 1>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  check.expect_almost_eq_const(
      %dim1,
      dense<[[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]]> : tensor<2x3xf32>
  ) : tensor<2x3xf32>

  %both_dims = "stablehlo.reverse"(%t1) {dimensions = array<i64: 0, 1>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  check.expect_almost_eq_const(
      %both_dims,
      dense<[[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]> : tensor<2x3xf32>
  ) : tensor<2x3xf32>
  return
}

// Regression test for https://github.com/iree-org/iree/issues/23637.
func.func @xla_reverse_after_slice() {
  %t1 = util.unfoldable_constant dense<[
      [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
      [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]]
  ]> : tensor<2x4x2xf32>
  %sliced = "stablehlo.slice"(%t1) {
      start_indices = array<i64: 0, 1, 0>,
      limit_indices = array<i64: 2, 3, 2>,
      strides = array<i64: 1, 1, 1>
  } : (tensor<2x4x2xf32>) -> tensor<2x2x2xf32>
  %reversed = "stablehlo.reverse"(%sliced) {dimensions = array<i64: 1>} : (tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
  check.expect_almost_eq_const(
      %reversed,
      dense<[
          [[5.0, 6.0], [3.0, 4.0]],
          [[13.0, 14.0], [11.0, 12.0]]
      ]> : tensor<2x2x2xf32>
  ) : tensor<2x2x2xf32>
  return
}
