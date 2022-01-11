func @scan_1d_dim0_inclusive_sum() {
  %input = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf32>

  %init = linalg.init_tensor [6] : tensor<6xf32>
  %c0 = arith.constant 0.0 : f32
  %0 = iree_linalg_ext.scan
         dimension(0) inclusive(true)
         ins(%input, %c0 : tensor<6xf32>, f32)
         outs(%init : tensor<6xf32>) {
           ^bb0(%arg0 : f32, %arg1 : f32):
             %sum = arith.addf %arg0, %arg1 : f32
             iree_linalg_ext.yield %sum : f32
         } -> tensor<6xf32>

  check.expect_almost_eq_const(
      %0,
      dense<[1.0, 3.0, 6.0, 10.0, 15.0, 21.0]> : tensor<6xf32>
  ) : tensor<6xf32>

  return
}

func @scan_1d_dim0_exclusive_sum() {
  %input = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf32>

  %init = linalg.init_tensor [6] : tensor<6xf32>
  %c0 = arith.constant 0.0 : f32
  %0 = iree_linalg_ext.scan
         dimension(0) inclusive(false)
         ins(%input, %c0 : tensor<6xf32>, f32)
         outs(%init : tensor<6xf32>) {
           ^bb0(%arg0 : f32, %arg1 : f32):
             %sum = arith.addf %arg0, %arg1 : f32
             iree_linalg_ext.yield %sum : f32
         } -> tensor<6xf32>

  check.expect_almost_eq_const(
      %0,
      dense<[0.0, 1.0, 3.0, 6.0, 10.0, 15.0]> : tensor<6xf32>
  ) : tensor<6xf32>

  return
}

func @scan_1d_dim0_inclusive_mul() {
  %input = util.unfoldable_constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>

  %init = linalg.init_tensor [6] : tensor<6xi32>
  %c0 = arith.constant 1 : i32
  %0 = iree_linalg_ext.scan
         dimension(0) inclusive(true)
         ins(%input, %c0 : tensor<6xi32>, i32)
         outs(%init : tensor<6xi32>) {
           ^bb0(%arg0 : i32, %arg1 : i32):
             %sum = arith.muli %arg0, %arg1 : i32
             iree_linalg_ext.yield %sum : i32
         } -> tensor<6xi32>

  check.expect_eq_const(
      %0,
      dense<[1, 2, 6, 24, 120, 720]> : tensor<6xi32>
  ) : tensor<6xi32>

  return
}

func @scan_2d_dim0_inclusive_sum() {
  %input = util.unfoldable_constant dense<[[1, 2, 3],
                                           [4, 5, 6]]> : tensor<2x3xi32>

  %init = linalg.init_tensor [2, 3] : tensor<2x3xi32>
  %c0 = arith.constant 0 : i32
  %0 = iree_linalg_ext.scan
         dimension(0) inclusive(true)
         ins(%input, %c0 : tensor<2x3xi32>, i32)
         outs(%init : tensor<2x3xi32>) {
           ^bb0(%arg0 : i32, %arg1 : i32):
             %sum = arith.addi %arg0, %arg1 : i32
             iree_linalg_ext.yield %sum : i32
         } -> tensor<2x3xi32>

  check.expect_eq_const(
      %0,
      dense<[[1, 2, 3], [5, 7, 9]]> : tensor<2x3xi32>
  ) : tensor<2x3xi32>

  return
}

func @scan_2d_dim1_inclusive_sum() {
  %input = util.unfoldable_constant dense<[[1, 2, 3],
                                           [4, 5, 6]]> : tensor<2x3xi32>

  %init = linalg.init_tensor [2, 3] : tensor<2x3xi32>
  %c0 = arith.constant 0 : i32
  %0 = iree_linalg_ext.scan
         dimension(1) inclusive(true)
         ins(%input, %c0 : tensor<2x3xi32>, i32)
         outs(%init : tensor<2x3xi32>) {
           ^bb0(%arg0 : i32, %arg1 : i32):
             %sum = arith.addi %arg0, %arg1 : i32
             iree_linalg_ext.yield %sum : i32
         } -> tensor<2x3xi32>

  check.expect_eq_const(
      %0,
      dense<[[1, 3, 6], [4, 9, 15]]> : tensor<2x3xi32>
  ) : tensor<2x3xi32>

  return
}
