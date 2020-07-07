func @gather_concat() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[
    [[05, 06, 07, 08]],
    [[09, 10, 11, 12]]]> : tensor<2x1x4xi32>
  %start_indices = iree.unfoldable_constant dense<0> : tensor<i64>
  %gath = "mhlo.gather"(%input, %start_indices) {
    dimension_numbers = {
      collapsed_slice_dims = dense<0> : tensor<1xi64>,
      index_vector_dim = 0 : i64,
      offset_dims = dense<[0, 1]> : tensor<2xi64>,
      start_index_map = dense<0> : tensor<1xi64>},
      slice_sizes = dense<[1, 1, 4]> : tensor<3xi64>
  } : (tensor<2x1x4xi32>, tensor<i64>) -> tensor<1x4xi32>
  %suffix = iree.unfoldable_constant dense<[[1, 2]]> : tensor<1x2xi32>
  %res = "mhlo.concatenate"(%gath, %suffix) {dimension = 1 : i64} : (tensor<1x4xi32>, tensor<1x2xi32>) -> tensor<1x6xi32>
  check.expect_eq_const(%res, dense<[[5, 6, 7, 8, 1, 2]]> : tensor<1x6xi32>) : tensor<1x6xi32>
  return
}
