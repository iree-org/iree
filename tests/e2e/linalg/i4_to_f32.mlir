#map = affine_map<(d0) -> (d0)>
func.func @i4_to_f32() {
  %input = util.unfoldable_constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi4>
  %0 = tensor.empty() : tensor<8xf32>
  %res = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
    ins(%input : tensor<8xi4>) outs(%0 : tensor<8xf32>) {
  ^bb0(%in: i4, %out: f32):
    %2 = arith.extui %in : i4 to i32
    %3 = arith.uitofp %2 : i32 to f32
    linalg.yield %3 : f32
  } -> tensor<8xf32>
  check.expect_eq_const(%res, dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]> : tensor<8xf32>) : tensor<8xf32>
  return
}
