func @dot_general_lower() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<[[[0.3, 0.5]]]> : tensor<1x1x2xf32>
  %rhs = iree.unfoldable_constant  dense<[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]> : tensor<2x3xf32>
  %res = "mhlo.dot_general"(%lhs, %rhs) {
      dot_dimension_numbers = {
          lhs_batching_dimensions = dense<[]> : tensor<0xi64>,
          lhs_contracting_dimensions = dense<2> : tensor<1xi64>,
          rhs_batching_dimensions = dense<[]> : tensor<0xi64>,
          rhs_contracting_dimensions = dense<0> : tensor<1xi64>
        },
        precision_config = ["DEFAULT", "DEFAULT"]
  } : (tensor<1x1x2xf32>, tensor<2x3xf32>) -> tensor<1x1x3xf32>
  check.expect_almost_eq_const(%res, dense<[[[0.23, 0.31, 0.39]]]> : tensor<1x1x3xf32>) : tensor<1x1x3xf32>
  return
}

func @dot_general_lower_swapped() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant  dense<[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]> : tensor<2x3xf32>
  %rhs = iree.unfoldable_constant dense<[[[0.3, 0.5]]]> : tensor<1x1x2xf32>
  %res = "mhlo.dot_general"(%lhs, %rhs) {
        dot_dimension_numbers = {
            lhs_batching_dimensions = dense<[]> : tensor<0xi64>,
            lhs_contracting_dimensions = dense<0> : tensor<1xi64>,
            rhs_batching_dimensions = dense<[]> : tensor<0xi64>,
            rhs_contracting_dimensions = dense<2> : tensor<1xi64>
        },
        precision_config = ["DEFAULT", "DEFAULT"]
  } : (tensor<2x3xf32>, tensor<1x1x2xf32>) -> tensor<3x1x1xf32>
  check.expect_almost_eq_const(%res, dense<[[[0.23]],[[0.31]],[[0.39]]]> : tensor<3x1x1xf32>) : tensor<3x1x1xf32>
  return
}

func @dot_general_trivial_batching_dimension() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant  dense<[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]> : tensor<1x2x3xf32>
  %rhs = iree.unfoldable_constant dense<[[
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0]]]> : tensor<1x3x4xf32>
  %res = "mhlo.dot_general"(%lhs, %rhs) {
        dot_dimension_numbers = {
            lhs_batching_dimensions = dense<0> : tensor<1xi64>,
            lhs_contracting_dimensions = dense<2> : tensor<1xi64>,
            rhs_batching_dimensions = dense<0> : tensor<1xi64>,
            rhs_contracting_dimensions = dense<1> : tensor<1xi64>
        },
        precision_config = ["DEFAULT", "DEFAULT"]
  } : (tensor<1x2x3xf32>, tensor<1x3x4xf32>) -> tensor<1x2x4xf32>
  check.expect_almost_eq_const(%res, dense<[[[0.6, 1.2, 1.8, 2.4],[1.5, 3.0, 4.5, 6.0]]]> : tensor<1x2x4xf32>) : tensor<1x2x4xf32>
  return
}

func @dot_general_nontrivial_batching_dimension() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<[
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]> : tensor<2x2x3xf32>
  %rhs = iree.unfoldable_constant dense<[[
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0]
  ], [
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0]]]> : tensor<2x3x4xf32>
  %res = "mhlo.dot_general"(%lhs, %rhs) {
        dot_dimension_numbers = {
            lhs_batching_dimensions = dense<0> : tensor<1xi64>,
            lhs_contracting_dimensions = dense<2> : tensor<1xi64>,
            rhs_batching_dimensions = dense<0> : tensor<1xi64>,
            rhs_contracting_dimensions = dense<1> : tensor<1xi64>
        },
        precision_config = ["DEFAULT", "DEFAULT"]
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
