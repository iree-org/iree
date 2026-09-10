func.func @reduce_window_nonoverlapping_1x4x6x1xf32() {
  %0 = util.unfoldable_constant dense<[[[[ 1.0], [ 2.0], [ 3.0], [ 4.0], [ 5.0], [ 6.0]],
                                        [[ 7.0], [ 8.0], [ 9.0], [10.0], [11.0], [12.0]],
                                        [[13.0], [14.0], [15.0], [16.0], [17.0], [18.0]],
                                        [[19.0], [20.0], [21.0], [22.0], [23.0], [24.0]]]]> : tensor<1x4x6x1xf32>
  %1 = util.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "stablehlo.reduce_window"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%3) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 2, 3, 1>,
      window_strides = array<i64: 1, 2, 3, 1>} : (tensor<1x4x6x1xf32>, tensor<f32>) -> tensor<1x2x2x1xf32>
  check.expect_eq_const(%res, dense<[[[[30.0], [48.0]],[[102.0], [120.0]]]]> : tensor<1x2x2x1xf32>) : tensor<1x2x2x1xf32>
  return
}

func.func @reduce_window_overlapping_4x6xf32() {
  %0 = util.unfoldable_constant dense<[[[[ 1.0], [ 2.0], [ 3.0], [ 4.0], [ 5.0], [ 6.0]],
                                        [[ 7.0], [ 8.0], [ 9.0], [10.0], [11.0], [12.0]],
                                        [[13.0], [14.0], [15.0], [16.0], [17.0], [18.0]],
                                        [[19.0], [20.0], [21.0], [22.0], [23.0], [24.0]]]]> : tensor<1x4x6x1xf32>
  %1 = util.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "stablehlo.reduce_window"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%3) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 2, 3, 1>,
      window_strides = array<i64: 1, 1, 1, 1>} : (tensor<1x4x6x1xf32>, tensor<f32>) -> tensor<1x3x4x1xf32>
  check.expect_eq_const(%res, dense<[[
      [[ 30.0], [ 36.0], [ 42.0], [ 48.0]],
      [[ 66.0], [ 72.0], [ 78.0], [ 84.0]],
      [[102.0], [108.0], [114.0], [120.0]]]]> : tensor<1x3x4x1xf32>) : tensor<1x3x4x1xf32>
  return
}

func.func @reduce_window_max_4x6xf32() {
  %0 = util.unfoldable_constant dense<[[[[ 1.0], [ 2.0], [ 3.0], [ 4.0], [ 5.0], [ 6.0]],
                                        [[ 7.0], [ 8.0], [ 9.0], [10.0], [11.0], [12.0]],
                                        [[13.0], [14.0], [15.0], [16.0], [17.0], [18.0]],
                                        [[19.0], [20.0], [21.0], [22.0], [23.0], [24.0]]]]> : tensor<1x4x6x1xf32>
  %1 = util.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "stablehlo.reduce_window"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %3 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%3) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 2, 3, 1>,
      window_strides = array<i64: 1, 2, 3, 1>} : (tensor<1x4x6x1xf32>, tensor<f32>) -> tensor<1x2x2x1xf32>
  check.expect_almost_eq_const(%res, dense<[[[[9.0], [12.0]], [[21.0], [24.0]]]]> : tensor<1x2x2x1xf32>) : tensor<1x2x2x1xf32>
  return
}

func.func @reduce_window_min_4x6xf32() {
  %0 = util.unfoldable_constant dense<[[[[ 1.0], [ 2.0], [ 3.0], [ 4.0], [ 5.0], [ 6.0]],
                                        [[ 7.0], [ 8.0], [ 9.0], [10.0], [11.0], [12.0]],
                                        [[13.0], [14.0], [15.0], [16.0], [17.0], [18.0]],
                                        [[19.0], [20.0], [21.0], [22.0], [23.0], [24.0]]]]> : tensor<1x4x6x1xf32>
  %1 = util.unfoldable_constant dense<14.0> : tensor<f32>
  %res = "stablehlo.reduce_window"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %3 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%3) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 2, 3, 1>,
      window_strides = array<i64: 1, 2, 3, 1>} : (tensor<1x4x6x1xf32>, tensor<f32>) -> tensor<1x2x2x1xf32>
  check.expect_almost_eq_const(%res, dense<[[[[1.0], [4.0]], [[13.0], [14.0]]]]> : tensor<1x2x2x1xf32>) : tensor<1x2x2x1xf32>
  return
}

func.func @reduce_window_max_with_padding_4x6xf32() {
  %0 = util.unfoldable_constant dense<[[[[ 1.0], [ 2.0], [ 3.0], [ 4.0], [ 5.0], [ 6.0]],
                                        [[ 7.0], [ 8.0], [ 9.0], [10.0], [11.0], [12.0]],
                                        [[13.0], [14.0], [15.0], [16.0], [17.0], [18.0]],
                                        [[19.0], [20.0], [21.0], [22.0], [23.0], [24.0]]]]> : tensor<1x4x6x1xf32>
  %1 = util.unfoldable_constant dense<0.0> : tensor<f32>
  %res = "stablehlo.reduce_window"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %3 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%3) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 2, 3, 1>,
      window_strides = array<i64: 1, 2, 3, 1>,
      padding = dense<[[0, 0], [1, 1], [0, 0], [0, 0]]> : tensor<4x2xi64>} : (tensor<1x4x6x1xf32>, tensor<f32>) -> tensor<1x3x2x1xf32>
  check.expect_almost_eq_const(%res, dense<[[[[3.0], [6.0]], [[15.0], [18.0]], [[21.0], [24.0]]]]> : tensor<1x3x2x1xf32>) : tensor<1x3x2x1xf32>
  return
}

func.func @cumsum_f32() {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = util.unfoldable_constant dense<1.0> : tensor<2x2x2xf32>
  %res = "stablehlo.reduce_window"(%1, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %4 = stablehlo.add %arg1, %arg2 : tensor<f32>
    "stablehlo.return"(%4) : (tensor<f32>) -> ()
  }) {padding = dense<[[1, 0], [0, 0], [0, 0]]> : tensor<3x2xi64>,
      window_dimensions = array<i64: 2, 1, 1>,
      window_strides = array<i64: 1, 1, 1>
  } : (tensor<2x2x2xf32>, tensor<f32>) -> tensor<2x2x2xf32>
  check.expect_almost_eq_const(%res, dense<[[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]]> : tensor<2x2x2xf32>) : tensor<2x2x2xf32>
  return
}

// Empty inputs must produce empty results.
func.func @dynamic_reduce_window_n0_w3_s2() {
  %a = flow.tensor.dynamic_constant dense<> : tensor<0xf32> -> tensor<?xf32>
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %r = "stablehlo.reduce_window"(%a, %init) ({
  ^bb0(%x: tensor<f32>, %y: tensor<f32>):
    %sum = stablehlo.add %x, %y : tensor<f32>
    "stablehlo.return"(%sum) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 3>, window_strides = array<i64: 2>} : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  %size = stablehlo.get_dimension_size %r, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  check.expect_eq_const(%size, dense<0> : tensor<i32>) : tensor<i32>
  return
}

// A window larger than the input must produce an empty result.
func.func @dynamic_reduce_window_n1_w3_s2() {
  %a = flow.tensor.dynamic_constant dense<[1.0]> : tensor<1xf32> -> tensor<?xf32>
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %r = "stablehlo.reduce_window"(%a, %init) ({
  ^bb0(%x: tensor<f32>, %y: tensor<f32>):
    %sum = stablehlo.add %x, %y : tensor<f32>
    "stablehlo.return"(%sum) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 3>, window_strides = array<i64: 2>} : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  %size = stablehlo.get_dimension_size %r, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  check.expect_eq_const(%size, dense<0> : tensor<i32>) : tensor<i32>
  return
}

// Truncating a negative span must not count a partial window.
func.func @dynamic_reduce_window_n2_w3_s2() {
  %a = flow.tensor.dynamic_constant dense<[1.0, 2.0]> : tensor<2xf32> -> tensor<?xf32>
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %r = "stablehlo.reduce_window"(%a, %init) ({
  ^bb0(%x: tensor<f32>, %y: tensor<f32>):
    %sum = stablehlo.add %x, %y : tensor<f32>
    "stablehlo.return"(%sum) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 3>, window_strides = array<i64: 2>} : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  %size = stablehlo.get_dimension_size %r, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  check.expect_eq_const(%size, dense<0> : tensor<i32>) : tensor<i32>
  return
}

// A window that exactly fits must produce one result.
func.func @dynamic_reduce_window_n3_w3_s2() {
  %a = flow.tensor.dynamic_constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32> -> tensor<?xf32>
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %r = "stablehlo.reduce_window"(%a, %init) ({
  ^bb0(%x: tensor<f32>, %y: tensor<f32>):
    %sum = stablehlo.add %x, %y : tensor<f32>
    "stablehlo.return"(%sum) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 3>, window_strides = array<i64: 2>} : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  %expected = arith.constant dense<[6.0]> : tensor<1xf32>
  %expected_dynamic = tensor.cast %expected : tensor<1xf32> to tensor<?xf32>
  check.expect_eq(%r, %expected_dynamic) : tensor<?xf32>
  return
}

// A trailing partial window must not contribute to the result.
func.func @dynamic_reduce_window_n6_w3_s2() {
  %a = flow.tensor.dynamic_constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]> : tensor<6xf32> -> tensor<?xf32>
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %r = "stablehlo.reduce_window"(%a, %init) ({
  ^bb0(%x: tensor<f32>, %y: tensor<f32>):
    %sum = stablehlo.add %x, %y : tensor<f32>
    "stablehlo.return"(%sum) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 3>, window_strides = array<i64: 2>} : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  %expected = arith.constant dense<[6.0, 12.0]> : tensor<2xf32>
  %expected_dynamic = tensor.cast %expected : tensor<2xf32> to tensor<?xf32>
  check.expect_eq(%r, %expected_dynamic) : tensor<?xf32>
  return
}

// A span below minus one stride must not produce a negative extent.
func.func @dynamic_reduce_window_n2_w5_s1() {
  %a = flow.tensor.dynamic_constant dense<[1.0, 2.0]> : tensor<2xf32> -> tensor<?xf32>
  %init = stablehlo.constant dense<0.0> : tensor<f32>
  %r = "stablehlo.reduce_window"(%a, %init) ({
  ^bb0(%x: tensor<f32>, %y: tensor<f32>):
    %sum = stablehlo.add %x, %y : tensor<f32>
    "stablehlo.return"(%sum) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 5>, window_strides = array<i64: 1>} : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  %size = stablehlo.get_dimension_size %r, dim = 0 : (tensor<?xf32>) -> tensor<i32>
  check.expect_eq_const(%size, dense<0> : tensor<i32>) : tensor<i32>
  return
}
