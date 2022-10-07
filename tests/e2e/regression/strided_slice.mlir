func.func @stride_slice() {
  %c15 = arith.constant 15 : i32
  %c16 = arith.constant 16 : i32
  %0 = tensor.empty() : tensor<12x15xi32>
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
  %2 = tensor.empty() : tensor<14x16xi32>
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
  %6 = tensor.empty() : tensor<3x3xi32>
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

#map = affine_map<(d0) -> (d0)>
func.func @issue_8825() {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c3_i64 = arith.constant 3 : i64
  %c2_i64 = arith.constant 2 : i64
  %arg0 = arith.constant dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf32>
  %0 = tensor.dim %arg0, %c0 : tensor<4xf32>
  %1 = arith.index_cast %0 : index to i64
  %2 = arith.addi %c2_i64, %1 : i64
  %3 = arith.cmpi sge, %c2_i64, %c0_i64 : i64
  %4 = arith.select %3, %c2_i64, %2 : i64
  %5 = arith.cmpi slt, %4, %c0_i64 : i64
  %6 = arith.select %5, %c0_i64, %4 : i64
  %7 = arith.cmpi sgt, %6, %1 : i64
  %8 = arith.select %7, %1, %6 : i64
  %9 = arith.index_cast %8 : i64 to index
  %10 = arith.cmpi sge, %0, %9 : index
  %11 = arith.select %10, %0, %9 : index
  %12 = arith.subi %11, %9 : index
  %13 = tensor.extract_slice %arg0[%9] [%12] [1] : tensor<4xf32> to tensor<?xf32>
  %14 = tensor.empty(%12) : tensor<?xf32>
  %16 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%13 : tensor<?xf32>) outs(%14 : tensor<?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %16 = arith.sitofp %c3_i64 : i64 to f32
      %17 = arith.mulf %arg1, %16 : f32
      linalg.yield %17 : f32
    } -> tensor<?xf32>
  %17 = tensor.cast %16 : tensor<?xf32> to tensor<2xf32>
  check.expect_almost_eq_const(%17, dense<[6.0, 9.0]> : tensor<2xf32>) : tensor<2xf32>
  return
}
