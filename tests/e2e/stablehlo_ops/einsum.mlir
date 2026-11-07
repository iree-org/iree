// Independent reductions multiply the separately reduced operands.
func.func @independent_reductions() {
  %lhs = util.unfoldable_constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %rhs = util.unfoldable_constant dense<[[5.0, 6.0], [7.0, 8.0]]> : tensor<2x2xf32>
  %r = stablehlo.einsum %lhs, %rhs, config = "ab,cd->ac" : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  check.expect_eq_const(%r, dense<[[33.0, 45.0], [77.0, 105.0]]> : tensor<2x2xf32>) : tensor<2x2xf32>
  return
}

func.func @scalar_independent_reduction() {
  %input = flow.tensor.dynamic_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32> -> tensor<?x?xf32>
  %scale = stablehlo.constant dense<2.0> : tensor<f32>
  %lhs = stablehlo.einsum %scale, %input, config = ",ab->a" : (tensor<f32>, tensor<?x?xf32>) -> tensor<?xf32>
  %rhs = stablehlo.einsum %input, %scale, config = "ab,->a" : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
  %lhs_fixed = tensor.cast %lhs : tensor<?xf32> to tensor<2xf32>
  %rhs_fixed = tensor.cast %rhs : tensor<?xf32> to tensor<2xf32>
  check.expect_eq_const(%lhs_fixed, dense<[12.0, 30.0]> : tensor<2xf32>) : tensor<2xf32>
  check.expect_eq(%lhs_fixed, %rhs_fixed) : tensor<2xf32>
  return
}

// Repeated labels read the diagonal before reducing, not every matrix entry.
func.func @diagonal_contraction() {
  %input = util.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]> : tensor<3x3xf32>
  %r = stablehlo.einsum %input, %input, config = "ii,ii->" : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<f32>
  check.expect_eq_const(%r, dense<107.0> : tensor<f32>) : tensor<f32>
  return
}

func.func @empty_independent_reductions() {
  %lhs = flow.tensor.dynamic_constant dense<> : tensor<2x0xf32> -> tensor<?x?xf32>
  %rhs = flow.tensor.dynamic_constant dense<> : tensor<3x0xf32> -> tensor<?x?xf32>
  %r = stablehlo.einsum %lhs, %rhs, config = "ab,cd->ac" : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %fixed = tensor.cast %r : tensor<?x?xf32> to tensor<2x3xf32>
  check.expect_eq_const(%fixed, dense<0.0> : tensor<2x3xf32>) : tensor<2x3xf32>
  return
}

func.func @dynamic_matvec() {
  %lhs = flow.tensor.dynamic_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32> -> tensor<?x?xf32>
  %rhs = flow.tensor.dynamic_constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32> -> tensor<?xf32>
  %r = stablehlo.einsum %lhs, %rhs, config = "ij,j->i" : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %fixed = tensor.cast %r : tensor<?xf32> to tensor<2xf32>
  check.expect_eq_const(%fixed, dense<[14.0, 32.0]> : tensor<2xf32>) : tensor<2xf32>
  return
}

// Equal extents make a positional pairing verifier-valid but numerically wrong.
func.func @reordered_batch_and_contraction_labels() {
  %lhs = util.unfoldable_constant dense<[[[[[1.0, 2.0], [3.0, 4.0]]], [[[5.0, 6.0], [7.0, 8.0]]]], [[[[9.0, 10.0], [11.0, 12.0]]], [[[13.0, 14.0], [15.0, 16.0]]]]]> : tensor<2x2x1x2x2xf32>
  %rhs = util.unfoldable_constant dense<[[[[[17.0, 18.0]], [[19.0, 20.0]]], [[[21.0, 22.0]], [[23.0, 24.0]]]], [[[[25.0, 26.0]], [[27.0, 28.0]]], [[[29.0, 30.0]], [[31.0, 32.0]]]]]> : tensor<2x2x2x1x2xf32>
  %r = stablehlo.einsum %lhs, %rhs, config = "bcijk,kcblj->bcil" : (tensor<2x2x1x2x2xf32>, tensor<2x2x2x1x2xf32>) -> tensor<2x2x1x1xf32>
  check.expect_eq_const(%r, dense<[[[[225.0]], [[673.0]]], [[[997.0]], [[1605.0]]]]> : tensor<2x2x1x1xf32>) : tensor<2x2x1x1xf32>
  return
}
