#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func.func @argmax_1d() {
  // Input :-
  //    1 0 3 0
  //    0 2 0 4
  %zero = arith.constant 0.0 : f16
  %input_init = tensor.empty() : tensor<2x4xf16>
  %input_filled = linalg.fill ins(%zero : f16) outs(%input_init : tensor<2x4xf16>) -> tensor<2x4xf16>
  %one = arith.constant 1.0 : f16
  %index_one = arith.constant 0 : index
  %two = arith.constant 2.0 : f16
  %index_two = arith.constant 1 : index
  %three = arith.constant 3.0 : f16
  %index_three = arith.constant 2 : index
  %four = arith.constant 4.0 : f16
  %index_four = arith.constant 3 : index
  %input_1 = tensor.insert %one into %input_filled[0][%index_one] : tensor<2x4xf16>
  %input_2 = tensor.insert %two into %input_1[1][%index_two] : tensor<2x4xf16>
  %input_3 = tensor.insert %three into %input_3[0][%index_three] : tensor<2x4xf16>
  %final_input = tensor.insert %four into %input_4[1][%index_four] : tensor<2x4xf16>
  // Indices :-
  //    0 1 0 1
  %indices = tensor.empty() : tensor<4xi64>
  %zero_index = arith.constant 0 : i64
  %one_index = arith.constant 1 : i64
  %indices_1 = tensor.insert %zero_index into %indices[0] : tensor<4xi64>
  %indices_2 = tensor.insert %one_index into %indices_1[1] : tensor<4xi64>
  %indices_3 = tensor.insert %zero_index into %indices_2[2] : tensor<4xi64>
  %final_indices = tensor.insert %one_index into %indices_3[3] : tensor<4xi64>
  // Empty tensor : 4x5 -> ALL ZEROES
  %empty = tensor.empty() : tensor<4x4xf16>
  %final_empty = linalg.fill ins(%zero : f16) outs(%empty : tensor<4x4xf16>) -> tensor<4x4xf16>
  // Gather-like op :-
  //    1 2 3 4
  //    1 2 3 4
  //    1 2 3 4
  //    1 2 3 4
  %res = linalg.generic {
    indexing_maps = [#map, #map1],
    iterator_types = ["parallel", "parallel"]
  } ins(%final_indices : tensor<4xi64>) outs(%final_empty : tensor<4x4xf16>) {
  ^bb0(%in: i64, %out: f16):
    %dim1 = linalg.index 1 : index
    %dim0 = arith.index_cast %in : i64 to index
    %extracted = tensor.extract %final_input[%dim0, %dim1] : tensor<2x4xf16>
    linalg.yield %extracted : f16
  } -> tensor<4x4xf16>

  check.expect_eq_const(%res, dense<[[1, 2, 3, 4],[1, 2, 3, 4],[1, 2, 3, 4],[1, 2, 3, 4]]> : tensor<4x4xf16>) : tensor<4x4xf16>

  return
}
