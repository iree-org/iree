func.func @select_with_binary() {
  %control = arith.constant dense<[true, false, true, false]> : tensor<4xi1>
  %a = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %b = arith.constant dense<[5, 6, 7, 8]> : tensor<4xi32>
  %init = tensor.empty() : tensor<4xi32>
  %c = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%control, %a, %b : tensor<4xi1>, tensor<4xi32>, tensor<4xi32>)
      outs(%init : tensor<4xi32>) {
    ^bb0(%b1 : i1, %b2 : i32, %b3 : i32, %b4 : i32):
      %0 = arith.select %b1, %b2, %b3 : i32
      linalg.yield %0 : i32
    } -> tensor<4xi32>
  check.expect_eq_const(%c, dense<[1, 6, 3, 8]> : tensor<4xi32>) : tensor<4xi32>
  return
}
