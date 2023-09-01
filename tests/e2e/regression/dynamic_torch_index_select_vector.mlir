func.func @torch_index_select1() {
  %lhs = flow.tensor.constant
    dense<[[[1, 2],[3, 4]],[[5, 6],[7, 8]],[[9, 10],[11, 12]]]> : tensor<3x2x2xi32> -> tensor<?x?x?xi32>
  %rhs = flow.tensor.constant dense<[0, 1]> : tensor<2xi32> -> tensor<?xi32>
  %0 = "stablehlo.torch_index_select"(%lhs, %rhs) {batch_dims = 0 : i64, dim = 1 : i64} : (tensor<?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?xi32>
  %dshape = util.optimization_barrier %0 : tensor<?x?x?xi32>
  %result = tensor.cast %dshape : tensor<?x?x?xi32> to tensor<3x2x2xi32>
  check.expect_eq_const(%result,
    dense<[[[1, 2],[3, 4]],
           [[5, 6],[7, 8]],
           [[9, 10],[11, 12]]]> : tensor<3x2x2xi32>) : tensor<3x2x2xi32>
  return
}

func.func @torch_index_select2() {
  %lhs = flow.tensor.constant
    dense<[[[1, 2],[3, 4]],[[5, 6],[7, 8]],[[9, 10],[11, 12]]]> : tensor<3x2x2xi32> -> tensor<?x?x?xi32>
  %rhs = flow.tensor.constant dense<[0, 1]> : tensor<2xi32> -> tensor<?xi32>
  %0 = "stablehlo.torch_index_select"(%lhs, %rhs) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x?x?xi32>, tensor<?xi32>) -> tensor<?x?x?xi32>
  %dshape = util.optimization_barrier %0 : tensor<?x?x?xi32>
  %result = tensor.cast %dshape : tensor<?x?x?xi32> to tensor<2x2x2xi32>
  check.expect_eq_const(%result,
    dense<[[[1, 2],[3, 4]],
           [[5, 6],[7, 8]]]> : tensor<2x2x2xi32>) : tensor<2x2x2xi32>
  return
}
