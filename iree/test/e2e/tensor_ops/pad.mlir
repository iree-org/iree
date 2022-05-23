func.func @pad_only() {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %0 = linalg.init_tensor [4, 5] : tensor<4x5xi32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> ()>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%c1 : i32) outs(%0 : tensor<4x5xi32>) {
      ^bb0(%b0: i32, %b1 : i32):
        linalg.yield %c1 : i32
    } -> tensor<4x5xi32>
  %2 = tensor.pad %1 low[2, 3] high[4, 5] {
    ^bb0(%arg0 : index, %arg1 : index):
      tensor.yield %c0 : i32
  } : tensor<4x5xi32> to tensor<10x13xi32>
  check.expect_eq_const(%2, dense<[
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  ]> : tensor<10x13xi32>) : tensor<10x13xi32>
  return
}