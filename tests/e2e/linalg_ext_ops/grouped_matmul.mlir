func.func @grouped_matmul_regular() {
  %input = util.unfoldable_constant dense<[[1.0], [2.0], [3.0]]> : tensor<3x1xf32>
  %weights = util.unfoldable_constant dense<[
      [[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
      [[3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]> : tensor<2x1x8xf32>
  %offsets = util.unfoldable_constant dense<[1, 3]> : tensor<2xi64>
  %row_base = arith.constant 0 : index
  %output = util.unfoldable_constant dense<0.0> : tensor<3x8xf32>
  %result = iree_linalg_ext.group_matmul ins(
      %input, %weights, %offsets, %row_base : tensor<3x1xf32>,
      tensor<2x1x8xf32>, tensor<2xi64>, index)
    outs(%output : tensor<3x8xf32>) -> tensor<3x8xf32>
  check.expect_almost_eq_const(%result,
      dense<[[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [6.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [9.0, 12.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]> : tensor<3x8xf32>)
      : tensor<3x8xf32>
  return
}

func.func @grouped_matmul_narrow_n() {
  %input = util.unfoldable_constant dense<[
      [1.0, 2.0],
      [3.0, 4.0],
      [5.0, 6.0]]> : tensor<3x2xf32>
  %weights = util.unfoldable_constant dense<[
      [[2.0], [1.0]],
      [[4.0], [3.0]]]> : tensor<2x2x1xf32>
  %offsets = util.unfoldable_constant dense<[1, 3]> : tensor<2xi64>
  %row_base = arith.constant 0 : index
  %output = util.unfoldable_constant dense<0.0> : tensor<3x1xf32>
  %result = iree_linalg_ext.group_matmul ins(
      %input, %weights, %offsets, %row_base : tensor<3x2xf32>,
      tensor<2x2x1xf32>, tensor<2xi64>, index)
    outs(%output : tensor<3x1xf32>) -> tensor<3x1xf32>
  check.expect_almost_eq_const(%result,
      dense<[[4.0], [24.0], [38.0]]> : tensor<3x1xf32>)
      : tensor<3x1xf32>
  return
}

func.func @grouped_matmul_dynamic_k_and_columns() {
  %input = flow.tensor.dynamic_constant dense<[
      [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]>
    : tensor<3x2xf32> -> tensor<3x?xf32>
  %weights = flow.tensor.dynamic_constant dense<[
      [[1.0, 2.0], [3.0, 4.0]],
      [[5.0, 6.0], [7.0, 8.0]]]>
    : tensor<2x2x2xf32> -> tensor<2x?x?xf32>
  %offsets = util.unfoldable_constant dense<[1, 3]> : tensor<2xi64>
  %row_base = arith.constant 0 : index
  %output = flow.tensor.dynamic_constant dense<99.0>
    : tensor<3x2xf32> -> tensor<3x?xf32>
  %result = iree_linalg_ext.group_matmul ins(
      %input, %weights, %offsets, %row_base : tensor<3x?xf32>,
      tensor<2x?x?xf32>, tensor<2xi64>, index)
    outs(%output : tensor<3x?xf32>) -> tensor<3x?xf32>
  // Keep the result shape dynamic until after the grouped matmul is lowered.
  %barrier = util.optimization_barrier %result : tensor<3x?xf32>
  %result_static = tensor.cast %barrier : tensor<3x?xf32> to tensor<3x2xf32>
  check.expect_almost_eq_const(%result_static,
      dense<[[7.0, 10.0], [43.0, 50.0], [67.0, 78.0]]> : tensor<3x2xf32>)
      : tensor<3x2xf32>
  return
}
