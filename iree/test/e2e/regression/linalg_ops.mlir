func @multi_result() {
  %input1 = iree.unfoldable_constant dense<[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12]]> : tensor<3x4xi32>
  %input2 = iree.unfoldable_constant dense<[
      [13, 14, 15, 16],
      [17, 18, 19, 20],
      [21, 22, 23, 24]]> : tensor<3x4xi32>
  %init = linalg.init_tensor [3, 4] : tensor<3x4xi32>
  %0:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%input1, %input2 : tensor<3x4xi32>, tensor<3x4xi32>)
      outs(%init, %init : tensor<3x4xi32>, tensor<3x4xi32>) {
      ^bb0(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) :
          %1 = addi %arg0, %arg1 : i32
          %2 = muli %arg0, %arg1 : i32
          linalg.yield %1, %2 : i32, i32
      } -> (tensor<3x4xi32>, tensor<3x4xi32>)
  check.expect_eq_const(%0#0, dense<[
      [14, 16, 18, 20],
      [22, 24, 26, 28],
      [30, 32, 34, 36]]> : tensor<3x4xi32>) : tensor<3x4xi32>
  check.expect_eq_const(%0#1, dense<[
      [13, 28, 45, 64],
      [85, 108, 133, 160],
      [189, 220, 253, 288]]> : tensor<3x4xi32>) : tensor<3x4xi32>
  return
}
