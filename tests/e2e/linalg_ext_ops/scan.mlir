func.func @scan_1d_dim0_inclusive_sum() {
  %input = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf32>

  %init = tensor.empty() : tensor<6xf32>
  %t0 = util.unfoldable_constant dense<0.0> : tensor<f32>
  %0:2 = iree_linalg_ext.scan
         dimension(0) inclusive(true)
         ins(%input : tensor<6xf32>)
         outs(%init, %t0 : tensor<6xf32>, tensor<f32>) {
           ^bb0(%arg0 : f32, %arg1 : f32):
             %sum = arith.addf %arg0, %arg1 : f32
             iree_linalg_ext.yield %sum : f32
         } -> tensor<6xf32>, tensor<f32>

  check.expect_almost_eq_const(
      %0#0,
      dense<[1.0, 3.0, 6.0, 10.0, 15.0, 21.0]> : tensor<6xf32>
  ) : tensor<6xf32>

  check.expect_almost_eq_const(
      %0#1,
      dense<21.0> : tensor<f32>
  ) : tensor<f32>

  return
}

func.func @scan_1d_dim0_exclusive_sum() {
  %input = util.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf32>

  %init = tensor.empty() : tensor<6xf32>
  %t0 = util.unfoldable_constant dense<10.0> : tensor<f32>
  %0:2 = iree_linalg_ext.scan
         dimension(0) inclusive(false)
         ins(%input : tensor<6xf32>)
         outs(%init, %t0 : tensor<6xf32>, tensor<f32>) {
           ^bb0(%arg0 : f32, %arg1 : f32):
             %sum = arith.addf %arg0, %arg1 : f32
             iree_linalg_ext.yield %sum : f32
         } -> tensor<6xf32>, tensor<f32>

  check.expect_almost_eq_const(
      %0#0,
      dense<[10.0, 11.0, 13.0, 16.0, 20.0, 25.0]> : tensor<6xf32>
  ) : tensor<6xf32>

  check.expect_almost_eq_const(
      %0#1,
      dense<25.0> : tensor<f32>
  ) : tensor<f32>

  return
}

func.func @scan_1d_dim0_inclusive_mul() {
  %input = util.unfoldable_constant dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>

  %init = tensor.empty() : tensor<6xi32>
  %t0 = util.unfoldable_constant dense<1> : tensor<i32>
  %0:2 = iree_linalg_ext.scan
         dimension(0) inclusive(true)
         ins(%input : tensor<6xi32>)
         outs(%init, %t0 : tensor<6xi32>, tensor<i32>) {
           ^bb0(%arg0 : i32, %arg1 : i32):
             %sum = arith.muli %arg0, %arg1 : i32
             iree_linalg_ext.yield %sum : i32
         } -> tensor<6xi32>, tensor<i32>

  check.expect_eq_const(
      %0#0,
      dense<[1, 2, 6, 24, 120, 720]> : tensor<6xi32>
  ) : tensor<6xi32>

  check.expect_eq_const(
      %0#1,
      dense<720> : tensor<i32>
  ) : tensor<i32>

  return
}

func.func @scan_2d_dim0_inclusive_sum() {
  %input = util.unfoldable_constant dense<[[1, 2, 3],
                                           [4, 5, 6]]> : tensor<2x3xi32>

  %init = tensor.empty() : tensor<2x3xi32>
  %t0 = util.unfoldable_constant dense<[0, 0, 0]> : tensor<3xi32>
  %0:2 = iree_linalg_ext.scan
         dimension(0) inclusive(true)
         ins(%input : tensor<2x3xi32>)
         outs(%init, %t0 : tensor<2x3xi32>, tensor<3xi32>) {
           ^bb0(%arg0 : i32, %arg1 : i32):
             %sum = arith.addi %arg0, %arg1 : i32
             iree_linalg_ext.yield %sum : i32
         } -> tensor<2x3xi32>, tensor<3xi32>

  check.expect_eq_const(
      %0#0,
      dense<[[1, 2, 3], [5, 7, 9]]> : tensor<2x3xi32>
  ) : tensor<2x3xi32>

  check.expect_eq_const(
      %0#1,
      dense<[5, 7, 9]> : tensor<3xi32>
  ) : tensor<3xi32>

  return
}

func.func @scan_2d_dim1_inclusive_sum() {
  %input = util.unfoldable_constant dense<[[1, 2, 3],
                                           [4, 5, 6]]> : tensor<2x3xi32>

  %init = tensor.empty() : tensor<2x3xi32>
  %t0 = util.unfoldable_constant dense<[0, 0]> : tensor<2xi32>
  %0:2 = iree_linalg_ext.scan
         dimension(1) inclusive(true)
         ins(%input : tensor<2x3xi32>)
         outs(%init, %t0 : tensor<2x3xi32>, tensor<2xi32>) {
           ^bb0(%arg0 : i32, %arg1 : i32):
             %sum = arith.addi %arg0, %arg1 : i32
             iree_linalg_ext.yield %sum : i32
         } -> tensor<2x3xi32>, tensor<2xi32>

  check.expect_eq_const(
      %0#0,
      dense<[[1, 3, 6], [4, 9, 15]]> : tensor<2x3xi32>
  ) : tensor<2x3xi32>

  check.expect_eq_const(
      %0#1,
      dense<[6, 15]> : tensor<2xi32>
  ) : tensor<2xi32>

  return
}
