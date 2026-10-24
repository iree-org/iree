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

// A strided window reifies the dynamic result length from its runtime input.
func.func @reduce_window_dynamic_stride() {
  %input = flow.tensor.dynamic_constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32> -> tensor<?xi32>
  %init = util.unfoldable_constant dense<0> : tensor<i32>
  %result = "stablehlo.reduce_window"(%input, %init) ({
    ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %sum : tensor<i32>
  }) {window_dimensions = array<i64: 2>, window_strides = array<i64: 2>} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  %expected = arith.constant dense<[3, 7]> : tensor<2xi32>
  %expected_dynamic = tensor.cast %expected : tensor<2xi32> to tensor<?xi32>
  check.expect_eq(%result, %expected_dynamic) : tensor<?xi32>
  return
}

func.func @reduce_window_dynamic_empty_result() {
  %input = flow.tensor.dynamic_constant dense<[7]> : tensor<1xi32> -> tensor<?xi32>
  %init = util.unfoldable_constant dense<0> : tensor<i32>
  %result = "stablehlo.reduce_window"(%input, %init) ({
    ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %sum : tensor<i32>
  }) {window_dimensions = array<i64: 3>} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  %expected = arith.constant dense<> : tensor<0xi32>
  %expected_dynamic = tensor.cast %expected : tensor<0xi32> to tensor<?xi32>
  check.expect_eq(%result, %expected_dynamic) : tensor<?xi32>
  return
}

func.func @reduce_window_dynamic_padding_base_dilation() {
  %input = flow.tensor.dynamic_constant dense<[1, 2, 3]> : tensor<3xi32> -> tensor<?xi32>
  %init = util.unfoldable_constant dense<0> : tensor<i32>
  %result = "stablehlo.reduce_window"(%input, %init) ({
    ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %sum : tensor<i32>
  }) {window_dimensions = array<i64: 3>, window_strides = array<i64: 2>, base_dilations = array<i64: 2>, padding = dense<[[1, 1]]> : tensor<1x2xi64>} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  %expected = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %expected_dynamic = tensor.cast %expected : tensor<3xi32> to tensor<?xi32>
  check.expect_eq(%result, %expected_dynamic) : tensor<?xi32>
  return
}

func.func @reduce_window_dynamic_window_dilation() {
  %input = flow.tensor.dynamic_constant dense<[1, 2, 3, 4, 5]> : tensor<5xi32> -> tensor<?xi32>
  %init = util.unfoldable_constant dense<0> : tensor<i32>
  %result = "stablehlo.reduce_window"(%input, %init) ({
    ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %sum : tensor<i32>
  }) {window_dimensions = array<i64: 2>, window_strides = array<i64: 1>, window_dilations = array<i64: 2>} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  %expected = arith.constant dense<[4, 6, 8]> : tensor<3xi32>
  %expected_dynamic = tensor.cast %expected : tensor<3xi32> to tensor<?xi32>
  check.expect_eq(%result, %expected_dynamic) : tensor<?xi32>
  return
}

func.func @reduce_window_dynamic_multi_result() {
  %input0 = flow.tensor.dynamic_constant dense<[1, 2, 3]> : tensor<3xi32> -> tensor<?xi32>
  %input1 = flow.tensor.dynamic_constant dense<[5, 4, 3]> : tensor<3xi32> -> tensor<?xi32>
  %init0 = util.unfoldable_constant dense<0> : tensor<i32>
  %init1 = util.unfoldable_constant dense<100> : tensor<i32>
  %result:2 = "stablehlo.reduce_window"(%input0, %input1, %init0, %init1) ({
    ^bb0(%acc0: tensor<i32>, %acc1: tensor<i32>, %value0: tensor<i32>, %value1: tensor<i32>):
      %sum = stablehlo.add %acc0, %value0 : tensor<i32>
      %difference = stablehlo.subtract %acc1, %value1 : tensor<i32>
      stablehlo.return %sum, %difference : tensor<i32>, tensor<i32>
  }) {window_dimensions = array<i64: 1>} : (tensor<?xi32>, tensor<?xi32>, tensor<i32>, tensor<i32>) -> (tensor<?xi32>, tensor<?xi32>)
  %expected_sum = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %expected_difference = arith.constant dense<[95, 96, 97]> : tensor<3xi32>
  %expected_sum_dynamic = tensor.cast %expected_sum : tensor<3xi32> to tensor<?xi32>
  %expected_difference_dynamic = tensor.cast %expected_difference : tensor<3xi32> to tensor<?xi32>
  check.expect_eq(%result#0, %expected_sum_dynamic) : tensor<?xi32>
  check.expect_eq(%result#1, %expected_difference_dynamic) : tensor<?xi32>
  return
}

func.func @reduce_window_dynamic_zero_span() {
  %input = flow.tensor.dynamic_constant dense<[1, 2, 3]> : tensor<3xi32> -> tensor<?xi32>
  %init = util.unfoldable_constant dense<0> : tensor<i32>
  %result = "stablehlo.reduce_window"(%input, %init) ({
    ^bb0(%lhs: tensor<i32>, %rhs: tensor<i32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<i32>
      stablehlo.return %sum : tensor<i32>
  }) {window_dimensions = array<i64: 3>} : (tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
  %expected = arith.constant dense<[6]> : tensor<1xi32>
  %expected_dynamic = tensor.cast %expected : tensor<1xi32> to tensor<?xi32>
  check.expect_eq(%result, %expected_dynamic) : tensor<?xi32>
  return
}

func.func @reduce_window_dynamic_pool_dilation() {
  %input = flow.tensor.dynamic_constant dense<1.0> : tensor<1x7x7x1xf32> -> tensor<1x?x?x1xf32>
  %init = util.unfoldable_constant dense<0.0> : tensor<f32>
  %result = "stablehlo.reduce_window"(%input, %init) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %max = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %max : tensor<f32>
  }) {window_dimensions = array<i64: 1, 2, 2, 1>, window_strides = array<i64: 1, 2, 2, 1>, window_dilations = array<i64: 1, 2, 2, 1>} : (tensor<1x?x?x1xf32>, tensor<f32>) -> tensor<1x?x?x1xf32>
  %expected = arith.constant dense<1.0> : tensor<1x3x3x1xf32>
  %expected_dynamic = tensor.cast %expected : tensor<1x3x3x1xf32> to tensor<1x?x?x1xf32>
  check.expect_eq(%result, %expected_dynamic) : tensor<1x?x?x1xf32>
  return
}

func.func @reduce_window_dynamic_pool_empty() {
  %input = flow.tensor.dynamic_constant dense<1.0> : tensor<1x1x1x1xf32> -> tensor<1x?x?x1xf32>
  %init = util.unfoldable_constant dense<0.0> : tensor<f32>
  %result = "stablehlo.reduce_window"(%input, %init) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %max = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %max : tensor<f32>
  }) {window_dimensions = array<i64: 1, 2, 2, 1>} : (tensor<1x?x?x1xf32>, tensor<f32>) -> tensor<1x?x?x1xf32>
  %expected = arith.constant dense<> : tensor<1x0x0x1xf32>
  %expected_dynamic = tensor.cast %expected : tensor<1x0x0x1xf32> to tensor<1x?x?x1xf32>
  check.expect_eq(%result, %expected_dynamic) : tensor<1x?x?x1xf32>
  return
}

func.func @reduce_window_dynamic_batch_max_2x4x3x1xf32() {
  %0 = flow.tensor.dynamic_constant dense<[[[[ 0.0], [ 1.0], [ 2.0]],
                                            [[ 3.0], [ 4.0], [ 5.0]],
                                            [[ 6.0], [ 7.0], [ 8.0]],
                                            [[ 9.0], [10.0], [11.0]]],
                                           [[[12.0], [13.0], [14.0]],
                                            [[15.0], [16.0], [17.0]],
                                            [[18.0], [19.0], [20.0]],
                                            [[21.0], [22.0], [23.0]]]]> : tensor<2x4x3x1xf32> -> tensor<?x4x3x1xf32>
  %1 = util.unfoldable_constant dense<0xFF800000> : tensor<f32>
  %res = "stablehlo.reduce_window"(%0, %1) ( {
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %3 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "stablehlo.return"(%3) : (tensor<f32>) -> ()
  }) {window_dimensions = array<i64: 1, 3, 2, 1>,
      window_strides = array<i64: 1, 1, 1, 1>} : (tensor<?x4x3x1xf32>, tensor<f32>) -> tensor<?x2x2x1xf32>
  %static = tensor.cast %res : tensor<?x2x2x1xf32> to tensor<2x2x2x1xf32>
  check.expect_eq_const(%static, dense<[[[[ 7.0], [ 8.0]], [[10.0], [11.0]]],
                                        [[[19.0], [20.0]], [[22.0], [23.0]]]]> : tensor<2x2x2x1xf32>) : tensor<2x2x2x1xf32>
  return
}
