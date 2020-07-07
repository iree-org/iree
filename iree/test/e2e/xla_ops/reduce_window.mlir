func @reduce_window_nonoverlapping_4x6xi32() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[ 1,  2,  3,  4,  5,  6],
                                       [ 7,  8,  9, 10, 11, 12],
                                       [13, 14, 15, 16, 17, 18],
                                       [19, 20, 21, 22, 23, 24]]> : tensor<4x6xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %res = "mhlo.reduce_window"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%3) : (tensor<i32>) -> ()
  }) {window_dimensions = dense<[2, 3]> : tensor<2xi64>,
      window_strides = dense<[2, 3]> : tensor<2xi64>} : (tensor<4x6xi32>, tensor<i32>) -> tensor<2x2xi32>
  check.expect_eq_const(%res, dense<[[30, 48],[102, 120]]> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

func @reduce_window_overlapping_4x6xi32() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[ 1,  2,  3,  4,  5,  6],
                                       [ 7,  8,  9, 10, 11, 12],
                                       [13, 14, 15, 16, 17, 18],
                                       [19, 20, 21, 22, 23, 24]]> : tensor<4x6xi32>
  %1 = iree.unfoldable_constant dense<0> : tensor<i32>
  %res = "mhlo.reduce_window"(%0, %1) ( {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):   // no predecessors
    %3 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%3) : (tensor<i32>) -> ()
  }) {window_dimensions = dense<[2, 3]> : tensor<2xi64>,
      window_strides = dense<[1, 1]> : tensor<2xi64>} : (tensor<4x6xi32>, tensor<i32>) -> tensor<3x4xi32>
  check.expect_eq_const(%res, dense<[
      [30, 36, 42, 48],
      [66, 72, 78, 84],
      [102, 108, 114, 120]]> : tensor<3x4xi32>) : tensor<3x4xi32>
  return
}

func @reduce_window_max_4x6xf32() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[ 1.0,  2.0,  3.0,  4.0,  5.0,  6.0],
                                       [ 7.0,  8.0,  9.0, 10.0, 11.0, 12.0],
                                       [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                                       [19.0, 20.0, 21.0, 22.0, 23.0, 24.0]]> : tensor<4x6xf32>
  %1 = iree.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "mhlo.reduce_window"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "mhlo.maximum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%3) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[2, 3]> : tensor<2xi64>,
      window_strides = dense<[2, 3]> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<2x2xf32>
  check.expect_almost_eq_const(%res, dense<[[9.0, 12.0], [21.0, 24.0]]> : tensor<2x2xf32>) : tensor<2x2xf32>
  return
}

func @reduce_window_min_4x6xf32() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[ -1.0,  -2.0,  -3.0,  -4.0,  -5.0,  -6.0],
                                       [ -7.0,  -8.0,  -9.0, -10.0, -11.0, -12.0],
                                       [-13.0, -14.0, -15.0, -16.0, -17.0, -18.0],
                                       [-19.0, -20.0, -21.0, -22.0, -23.0, -24.0]]> : tensor<4x6xf32>
  %1 = iree.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "mhlo.reduce_window"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):   // no predecessors
    %3 = "mhlo.minimum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%3) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[2, 3]> : tensor<2xi64>,
      window_strides = dense<[2, 3]> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<2x2xf32>
  check.expect_almost_eq_const(%res, dense<[[-9.0, -12.0], [-21.0, -24.0]]> : tensor<2x2xf32>) : tensor<2x2xf32>
  return
}
