func.func @torch_index_select1() {
  %lhs = flow.tensor.constant
    dense<[[[1,2,3,4,5]],
           [[6,7,8,9,10]],
           [[11,12,13,14,15]],
           [[16,17,18,19,20]],
           [[21,22,23,24,25]]]> : tensor<5x1x5xi32> -> tensor<?x?x?xi32>
  %rhs = util.unfoldable_constant dense<0> : tensor<i32>
  %0 = "stablehlo.torch_index_select"(%lhs, %rhs) {batch_dims = 0 : i64, dim = 0 : i64} : (tensor<?x?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  %dshape = util.optimization_barrier %0 : tensor<?x?xi32>
  %result = tensor.cast %dshape : tensor<?x?xi32> to tensor<1x5xi32>
  check.expect_eq_const(%result,
    dense<[[1, 2, 3, 4, 5]]> : tensor<1x5xi32>) : tensor<1x5xi32>
  return
}

func.func @torch_index_select2() {
   %lhs = flow.tensor.constant
    dense<[[[1,2,3,4,5]],
           [[6,7,8,9,10]],
           [[11,12,13,14,15]],
           [[16,17,18,19,20]],
           [[21,22,23,24,25]]]> : tensor<5x1x5xi32> -> tensor<?x?x?xi32>
  %rhs = util.unfoldable_constant dense<0> : tensor<i32>
  %0 = "stablehlo.torch_index_select"(%lhs, %rhs) {batch_dims = 0 : i64, dim = 1 : i64} : (tensor<?x?x?xi32>, tensor<i32>) -> tensor<?x?xi32>
  %dshape = util.optimization_barrier %0 : tensor<?x?xi32>
  %result = tensor.cast %dshape : tensor<?x?xi32> to tensor<5x5xi32>
  check.expect_eq_const(%result,
    dense<[[1, 2, 3, 4, 5],
           [6, 7, 8, 9, 10],
           [11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20],
           [21, 22, 23, 24, 25]]> : tensor<5x5xi32>) : tensor<5x5xi32>
  return
}
