// Regression test for https://github.com/iree-org/iree/issues/24342.
//
// It fails on risc-v only because it does not apply the upstream pattern that
// recovers the logical indices of a `vector.gather` op.
// See https://github.com/iree-org/iree/commit/25dcacfcbcd4ec91260920d7296ad558aa03ef84
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
  %reversed = "stablehlo.reverse"(%sliced) {dimensions = array<i64: 1>}
      : (tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
  check.expect_almost_eq_const(
      %reversed,
      dense<[
          [[5.0, 6.0], [3.0, 4.0]],
          [[13.0, 14.0], [11.0, 12.0]]
      ]> : tensor<2x2x2xf32>
  ) : tensor<2x2x2xf32>
  return
}
