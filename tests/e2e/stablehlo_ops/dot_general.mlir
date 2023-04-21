func.func @dot_general_lower() {
  %lhs = util.unfoldable_constant dense<[[[0.3, 0.5]]]> : tensor<1x1x2xf32>
  %rhs = util.unfoldable_constant  dense<[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]> : tensor<2x3xf32>
  %res = "stablehlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [],
      rhs_contracting_dimensions = [0],
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x1x2xf32>, tensor<2x3xf32>) -> tensor<1x1x3xf32>
  check.expect_almost_eq_const(%res, dense<[[[0.23, 0.31, 0.39]]]> : tensor<1x1x3xf32>) : tensor<1x1x3xf32>
  return
}

func.func @dot_general_lower_swapped() {
  %lhs = util.unfoldable_constant  dense<[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]> : tensor<2x3xf32>
  %rhs = util.unfoldable_constant dense<[[[0.3, 0.5]]]> : tensor<1x1x2xf32>
  %res = "stablehlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      lhs_contracting_dimensions = [0],
      rhs_batching_dimensions = [],
      rhs_contracting_dimensions = [2],
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x3xf32>, tensor<1x1x2xf32>) -> tensor<3x1x1xf32>
  check.expect_almost_eq_const(%res, dense<[[[0.23]],[[0.31]],[[0.39]]]> : tensor<3x1x1xf32>) : tensor<3x1x1xf32>
  return
}

func.func @dot_general_trivial_batching_dimension() {
  %lhs = util.unfoldable_constant  dense<[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]> : tensor<1x2x3xf32>
  %rhs = util.unfoldable_constant dense<[[
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0]]]> : tensor<1x3x4xf32>
  %res = "stablehlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1],
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x2x3xf32>, tensor<1x3x4xf32>) -> tensor<1x2x4xf32>
  check.expect_almost_eq_const(%res, dense<[[[0.6, 1.2, 1.8, 2.4],[1.5, 3.0, 4.5, 6.0]]]> : tensor<1x2x4xf32>) : tensor<1x2x4xf32>
  return
}

func.func @dot_general_matmul() {
  %lhs = util.unfoldable_constant dense<3.0> : tensor<2x4xf32>
  %rhs = util.unfoldable_constant dense<2.0> : tensor<4x2xf32>
  %res = "stablehlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_batching_dimensions = [],
      rhs_contracting_dimensions = [0],
    >
  }  : (tensor<2x4xf32>, tensor<4x2xf32>) -> tensor<2x2xf32>
  check.expect_eq_const(%res, dense<24.0> : tensor<2x2xf32>) : tensor<2x2xf32>
  return
}

func.func @dot_general_matmul_i32.i32.i32() {
  %lhs = util.unfoldable_constant dense<3> : tensor<2x4xi32>
  %rhs = util.unfoldable_constant dense<2> : tensor<4x2xi32>
  %res = "stablehlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_batching_dimensions = [],
      rhs_contracting_dimensions = [0],
    >
  } : (tensor<2x4xi32>, tensor<4x2xi32>) -> tensor<2x2xi32>
  check.expect_eq_const(%res, dense<24> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

func.func @dot_general_nontrivial_batching_dimension() {
  %lhs = util.unfoldable_constant dense<[
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]> : tensor<2x2x3xf32>
  %rhs = util.unfoldable_constant dense<[[
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0]
  ], [
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0]]]> : tensor<2x3x4xf32>
  %res = "stablehlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1],
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x2x3xf32>, tensor<2x3x4xf32>) -> tensor<2x2x4xf32>
  check.expect_almost_eq_const(%res, dense<[
      [
          [0.6, 1.2, 1.8, 2.4],
          [1.5, 3.0, 4.5, 6.0]
      ], [
          [6.0, 12.0, 18.0, 24.0],
          [15.0, 30.0, 45.0, 60.0]]]> : tensor<2x2x4xf32>) : tensor<2x2x4xf32>
  return
}

func.func @large_dot_general() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<4x8x128xf32>
  %rhs = util.unfoldable_constant dense<0.4> : tensor<4x128x16xf32>
  %res = "stablehlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1],
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<4x8x128xf32>, tensor<4x128x16xf32>) -> tensor<4x8x16xf32>
  check.expect_almost_eq_const(%res, dense<51.2> : tensor<4x8x16xf32>) : tensor<4x8x16xf32>
  return
}

func.func @dot_general_nontrivial_batching_mutliple_parallel_dimension() {
  %lhs = util.unfoldable_constant dense<[
    [[[0.0], [1.0]], [[2.0], [3.0]], [[ 4.0], [ 5.0]]],
    [[[6.0], [7.0]], [[8.0], [9.0]], [[10.0], [11.0]]]
  ]> : tensor<2x3x2x1xf32>
  %rhs = util.unfoldable_constant dense<[
    [[0.0], [1.0]], [[2.0], [3.0]]
  ]> : tensor<2x2x1xf32>
  %res = "stablehlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [2],
      rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2]
    >,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<2x3x2x1xf32>, tensor<2x2x1xf32>) -> tensor<2x2x3x2xf32>
  check.expect_almost_eq_const(%res, dense<[
    [
      [[0.0,  0.0], [0.0,  4.0], [0.0,  8.0]],
      [[0.0, 12.0], [0.0, 16.0], [0.0, 20.0]]
    ],
    [
      [[1.0,  3.0], [3.0,  9.0], [ 5.0, 15.0]],
      [[7.0, 21.0], [9.0, 27.0], [11.0, 33.0]]
    ]
  ]> : tensor<2x2x3x2xf32>) : tensor<2x2x3x2xf32>
  return
}
