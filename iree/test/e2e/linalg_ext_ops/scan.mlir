func @scan_1d_dim0_inclusive_sum() {
  %input = util.unfoldable_constant dense<1.0> : tensor<32xf32>

  %init = linalg.init_tensor [32] : tensor<32xf32>
  %t0 = util.unfoldable_constant dense<0.0> : tensor<f32>
  %0:2 = iree_linalg_ext.scan
         dimension(0) inclusive(true)
         ins(%input : tensor<32xf32>)
         outs(%init, %t0 : tensor<32xf32>, tensor<f32>) {
           ^bb0(%arg0 : f32, %arg1 : f32):
             %sum = arith.addf %arg0, %arg1 : f32
             iree_linalg_ext.yield %sum : f32
         } -> tensor<32xf32>, tensor<f32>

  check.expect_almost_eq_const(
      %0#0,
      dense<[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
        14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
        26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0
      ]> : tensor<32xf32>
  ) : tensor<32xf32>

  check.expect_almost_eq_const(
      %0#1,
      dense<32.0> : tensor<f32>
  ) : tensor<f32>

  return
}

func @scan_1d_dim0_exclusive_sum() {
  %input = util.unfoldable_constant dense<1.0> : tensor<32xf32>

  %init = linalg.init_tensor [32] : tensor<32xf32>
  %t0 = util.unfoldable_constant dense<10.0> : tensor<f32>
  %0:2 = iree_linalg_ext.scan
         dimension(0) inclusive(false)
         ins(%input : tensor<32xf32>)
         outs(%init, %t0 : tensor<32xf32>, tensor<f32>) {
           ^bb0(%arg0 : f32, %arg1 : f32):
             %sum = arith.addf %arg0, %arg1 : f32
             iree_linalg_ext.yield %sum : f32
         } -> tensor<32xf32>, tensor<f32>

  check.expect_almost_eq_const(
      %0#0,
      dense<[
        10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
        22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0,
        34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0
      ]> : tensor<32xf32>
  ) : tensor<32xf32>

  check.expect_almost_eq_const(
      %0#1,
      dense<41.0> : tensor<f32>
  ) : tensor<f32>

  return
}

func @scan_1d_dim0_exclusive_mul() {
  %input = util.unfoldable_constant dense<2> : tensor<32xi32>

  %init = linalg.init_tensor [32] : tensor<32xi32>
  %t0 = util.unfoldable_constant dense<1> : tensor<i32>
  %0:2 = iree_linalg_ext.scan
         dimension(0) inclusive(false)
         ins(%input : tensor<32xi32>)
         outs(%init, %t0 : tensor<32xi32>, tensor<i32>) {
           ^bb0(%arg0 : i32, %arg1 : i32):
             %sum = arith.muli %arg0, %arg1 : i32
             iree_linalg_ext.yield %sum : i32
         } -> tensor<32xi32>, tensor<i32>

  check.expect_eq_const(
      %0#0,
      dense<[
        0x1,0x2,0x4,0x8,0x10,0x20,0x40,0x80,0x100,0x200,0x400,0x800,0x1000,
        0x2000, 0x4000,0x8000,0x10000,0x20000,0x40000,0x80000,0x100000,0x200000,
        0x400000,0x800000,0x1000000,0x2000000,0x4000000,0x8000000,0x10000000,
        0x20000000,0x40000000,0x80000000
      ]> : tensor<32xi32>
  ) : tensor<32xi32>

  check.expect_eq_const(
      %0#1,
      dense<0x80000000> : tensor<i32>
  ) : tensor<i32>

  return
}

func @scan_2d_dim0_inclusive_sum() {
  %input = util.unfoldable_constant dense<1> : tensor<32x3xi32>

  %init = linalg.init_tensor [32, 3] : tensor<32x3xi32>
  %t0 = util.unfoldable_constant dense<[0, 0, 0]> : tensor<3xi32>
  %0:2 = iree_linalg_ext.scan
         dimension(0) inclusive(true)
         ins(%input : tensor<32x3xi32>)
         outs(%init, %t0 : tensor<32x3xi32>, tensor<3xi32>) {
           ^bb0(%arg0 : i32, %arg1 : i32):
             %sum = arith.addi %arg0, %arg1 : i32
             iree_linalg_ext.yield %sum : i32
         } -> tensor<32x3xi32>, tensor<3xi32>

  check.expect_eq_const(
      %0#0,
      dense<[
        [ 1,  1,  1],
        [ 2,  2,  2],
        [ 3,  3,  3],
        [ 4,  4,  4],
        [ 5,  5,  5],
        [ 6,  6,  6],
        [ 7,  7,  7],
        [ 8,  8,  8],
        [ 9,  9,  9],
        [10, 10, 10],
        [11, 11, 11],
        [12, 12, 12],
        [13, 13, 13],
        [14, 14, 14],
        [15, 15, 15],
        [16, 16, 16],
        [17, 17, 17],
        [18, 18, 18],
        [19, 19, 19],
        [20, 20, 20],
        [21, 21, 21],
        [22, 22, 22],
        [23, 23, 23],
        [24, 24, 24],
        [25, 25, 25],
        [26, 26, 26],
        [27, 27, 27],
        [28, 28, 28],
        [29, 29, 29],
        [30, 30, 30],
        [31, 31, 31],
        [32, 32, 32]
      ]> : tensor<32x3xi32>
  ) : tensor<32x3xi32>

  check.expect_eq_const(
      %0#1,
      dense<32> : tensor<3xi32>
  ) : tensor<3xi32>

  return
}

//func @scan_2d_dim1_inclusive_sum() {
//  %input = util.unfoldable_constant dense<1> : tensor<2x32xi32>
//
//  %init = linalg.init_tensor [2, 32] : tensor<2x32xi32>
//  %t0 = util.unfoldable_constant dense<[0, 0]> : tensor<2xi32>
//  %0:2 = iree_linalg_ext.scan
//         dimension(1) inclusive(true)
//         ins(%input : tensor<2x32xi32>)
//         outs(%init, %t0 : tensor<2x32xi32>, tensor<2xi32>) {
//           ^bb0(%arg0 : i32, %arg1 : i32):
//             %sum = arith.addi %arg0, %arg1 : i32
//             iree_linalg_ext.yield %sum : i32
//         } -> tensor<2x32xi32>, tensor<2xi32>
//
//  check.expect_eq_const(
//      %0#0,
//      dense<[
//        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
//        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
//        26, 27, 28, 29, 30, 31, 32],
//        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
//        14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
//        26, 27, 28, 29, 30, 31, 32]
//      ]> : tensor<2x32xi32>
//  ) : tensor<2x32xi32>
//
//  check.expect_eq_const(
//      %0#1,
//      dense<32> : tensor<2xi32>
//  ) : tensor<2xi32>
//
//  return
//}
