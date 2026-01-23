#map = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @gather_like_op() {
  // Input :-
  //    1 0 3 0
  //    0 2 0 4
  %input_init = util.unfoldable_constant dense<[[1, 0, 3, 0],[0, 2, 0, 4]]> : tensor<2x4xi64>
  // Indices :-
  //    0 1 0 1
  %indices = util.unfoldable_constant dense<[0, 1, 0, 1]> : tensor<4xi64>
  %empty = tensor.empty() : tensor<4x4xi64>
  // Gather-like op should result in:-
  //    1 2 3 4
  //    1 2 3 4
  //    1 2 3 4
  //    1 2 3 4
  %res = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]
  } ins(%indices : tensor<4xi64>) outs(%empty : tensor<4x4xi64>) {
  ^bb0(%in: i64, %out: i64):
    %dim1 = linalg.index 1 : index
    %dim0 = arith.index_cast %in : i64 to index
    %extracted = tensor.extract %input_init[%dim0, %dim1] : tensor<2x4xi64>
    linalg.yield %extracted : i64
  } -> tensor<4x4xi64>

  check.expect_eq_const(%res, dense<[[1, 2, 3, 4],
                                     [1, 2, 3, 4],
                                     [1, 2, 3, 4],
                                     [1, 2, 3, 4]]> : tensor<4x4xi64>) : tensor<4x4xi64>

  return
}
