func.func @dot_general_trivial_batching_dimension() {
  %lhs = util.unfoldable_constant  dense<[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]> : tensor<1x2x3xf32>
  %rhs = util.unfoldable_constant dense<[[
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0],
    [1.0, 2.0, 3.0, 4.0]]]> : tensor<1x3x4xf32>
  %res = "mhlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1],
    >,
    precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]
  } : (tensor<1x2x3xf32>, tensor<1x3x4xf32>) -> tensor<1x2x4xf32>
  check.expect_almost_eq_const(%res, dense<[[[0.6, 1.2, 1.8, 2.4],[1.5, 3.0, 4.5, 6.0]]]> : tensor<1x2x4xf32>) : tensor<1x2x4xf32>
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
  %res = "mhlo.dot_general"(%lhs, %rhs) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1],
    >,
    precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]
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

func.func @large_dot_general2() {
  %lhs = util.unfoldable_constant dense<1.0> : tensor<4x32x1024xf32>
  %rhs = util.unfoldable_constant dense<0.4> : tensor<4x1024x64xf32>
  %res = "mhlo.dot_general"(%lhs, %rhs) {
           dot_dimension_numbers = #mhlo.dot<
             lhs_batching_dimensions = [0],
             lhs_contracting_dimensions = [2],
             rhs_batching_dimensions = [0],
             rhs_contracting_dimensions = [1],
           >,
           precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]
         } : (tensor<4x32x1024xf32>, tensor<4x1024x64xf32>) -> tensor<4x32x64xf32>
  check.expect_almost_eq_const(%res, dense<409.596> : tensor<4x32x64xf32>) : tensor<4x32x64xf32>
  return
}
