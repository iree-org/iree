func.func @torch_index_select1() {
  %lhs = flow.tensor.constant  dense<[[6,7],[8,9]]> : tensor<2x2xi32> -> tensor<?x?xi32>
  %rhs = flow.tensor.constant  dense<[[[[0,1],[1,0]],[[0,0],[1,1]]],[[[1,1],[0,0]],[[0,1],[1,0]]]]> : tensor<2x2x2x2xi32> -> tensor<?x?x?x?xi32>
  %0 = "stablehlo.torch_index_select"(%lhs, %rhs) {batch_dims = 1 : i64, dim = 1 : i64} : (tensor<?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  %dshape = util.optimization_barrier %0 : tensor<?x?x?x?xi32>
  %result = tensor.cast %dshape : tensor<?x?x?x?xi32> to tensor<2x2x2x2xi32>
  check.expect_eq_const(%result,
    dense<[[[[6, 7],[7, 6]],
            [[6, 6],[7, 7]]],
           [[[9, 9],[8, 8]],
            [[8, 9],[9, 8]]]]> : tensor<2x2x2x2xi32>) : tensor<2x2x2x2xi32>
  return
}

func.func @torch_index_select2() {
  %lhs = flow.tensor.constant  dense<[[6,7],[8,9]]> : tensor<2x2xi32> -> tensor<?x?xi32>
  %rhs = flow.tensor.constant  dense<[[[[0,1],[1,0]],[[0,0],[1,1]]],[[[1,1],[0,0]],[[0,1],[1,0]]]]> : tensor<2x2x2x2xi32> -> tensor<?x?x?x?xi32>
  %0 = "stablehlo.torch_index_select"(%lhs, %rhs) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x?xi32>, tensor<?x?x?x?xi32>) -> tensor<?x?x?x?x?xi32>
  %dshape = util.optimization_barrier %0 : tensor<?x?x?x?x?xi32>
  %result = tensor.cast %dshape : tensor<?x?x?x?x?xi32> to tensor<2x2x2x2x2xi32>
  check.expect_eq_const(%result,
    dense<[[[[[6, 7],[8, 9]],
             [[8, 9],[6, 7]]],
            [[[6, 7],[6, 7]],
             [[8, 9],[8, 9]]]],
           [[[[8, 9],[8, 9]],
             [[6, 7],[6, 7]]],
            [[[6, 7],[8, 9]],
             [[8, 9],[6, 7]]]]]> : tensor<2x2x2x2x2xi32>) : tensor<2x2x2x2x2xi32>
  return
}
