func.func @torch_index_select1() {
  %lhs = flow.tensor.constant  dense<[[[100, 101],[110, 111]],[[200, 201],[210, 211]]]> : tensor<2x2x2xi32> -> tensor<?x?x?xi32>
  %rhs = flow.tensor.constant  dense<[[[0, 1],[1, 0]],[[0, 0],[1, 1]]]> : tensor<2x2x2xi32> -> tensor<?x?x?xi32>
  %0 = "stablehlo.torch_index_select"(%lhs, %rhs) {batch_dims = -1 : i64, dim = -1 : i64} : (tensor<?x?x?xi32>, tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %dshape = util.optimization_barrier %0 : tensor<?x?x?xi32>
  %result = tensor.cast %dshape : tensor<?x?x?xi32> to tensor<2x2x2xi32>
  check.expect_eq_const(%result,
    dense<[[[100, 101],[111, 110]],
           [[200, 200],[211, 211]]]> : tensor<2x2x2xi32>) : tensor<2x2x2xi32>
  return
}
