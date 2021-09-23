func @foo() {
  %input = util.unfoldable_constant dense<[
    [[01, 02, 03, 04, 05]],
    [[06, 07, 08, 09, 10]],
    [[11, 12, 13, 14, 15]],
    [[16, 17, 18, 19, 20]],
    [[21, 22, 23, 24, 25]]]> : tensor<5x1x5xi32>
  %start_indices = util.unfoldable_constant dense<2> : tensor<i64>
  %res = "mhlo.gather"(%input, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0],
      index_vector_dim = 0,
      offset_dims = [0, 1],
      start_index_map = [0],
    >,
    slice_sizes = dense<[1, 1, 5]> : tensor<3xi64>
  } : (tensor<5x1x5xi32>, tensor<i64>) -> tensor<1x5xi32>
  check.expect_eq_const(%res, dense<[[11, 12, 13, 14, 15]]> : tensor<1x5xi32>) : tensor<1x5xi32>
  return
}
