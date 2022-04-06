func @stride_slice() {
  %c15 = arith.constant 15 : i32
  %c16 = arith.constant 16 : i32
  %0 = linalg.init_tensor [12, 15] : tensor<12x15xi32>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%0 : tensor<12x15xi32>) {
    ^bb0(%b0 : i32):
      %2 = linalg.index 0 : index
      %3 = linalg.index 1 : index
      %4 = arith.index_cast %2 : index to i32
      %5 = arith.index_cast %3 : index to i32
      %6 = arith.muli %c15, %4 : i32
      %7 = arith.addi %6, %5 : i32
      linalg.yield %7 : i32
    } -> tensor<12x15xi32>
  %2 = linalg.init_tensor [14, 16] : tensor<14x16xi32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      outs(%2 : tensor<14x16xi32>) {
    ^bb0(%b0 : i32):
      %4 = linalg.index 0 : index
      %5 = linalg.index 1 : index
      %6 = arith.index_cast %4 : index to i32
      %7 = arith.index_cast %5 : index to i32
      %8 = arith.muli %c16, %6 : i32
      %9 = arith.addi %8, %7 : i32
      linalg.yield %9 : i32
    } -> tensor<14x16xi32>
  %4 = tensor.extract_slice %1[2, 3] [3, 3] [2, 3] : tensor<12x15xi32> to tensor<3x3xi32>
  %5 = tensor.extract_slice %3[3, 2] [3, 3] [3, 2] : tensor<14x16xi32> to tensor<3x3xi32>
  %6 = linalg.init_tensor [3, 3] : tensor<3x3xi32>
  %7 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%4, %5 : tensor<3x3xi32>, tensor<3x3xi32>)
      outs(%6 : tensor<3x3xi32>) {
    ^bb0(%b0 : i32, %b1 : i32, %b2: i32):
      %8 = arith.addi %b0, %b1 : i32
      linalg.yield %8 : i32
    } -> tensor<3x3xi32>
  %8 = arith.constant dense<42> : tensor<10x12xi32>
  %9 = tensor.insert_slice %7 into %8[1, 2] [3, 3] [2, 3] : tensor<3x3xi32> into tensor<10x12xi32>
  check.expect_eq_const(%9, dense<[
    [42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
    [42, 42, 83, 42, 42, 88, 42, 42, 93, 42, 42, 42],
    [42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
    [42, 42, 161, 42, 42, 166, 42, 42, 171, 42, 42, 42],
    [42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
    [42, 42, 239, 42, 42, 244, 42, 42, 249, 42, 42, 42],
    [42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
    [42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
    [42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42],
    [42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42]]> : tensor<10x12xi32>) : tensor<10x12xi32>
  return
}
