func.func @reverse_dim0() {
  %input = util.unfoldable_constant dense<[[1.0, 2.0, 3.0],
                                           [4.0, 5.0, 6.0]]> : tensor<2x3xf32>

  %init = tensor.empty() : tensor<2x3xf32>
  %0 = iree_linalg_ext.reverse
         dimensions(dense<0> : tensor<1xi64>)
         ins(%input : tensor<2x3xf32>)
         outs(%init : tensor<2x3xf32>) : tensor<2x3xf32>

  check.expect_almost_eq_const(
      %0,
      dense<[[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]]> : tensor<2x3xf32>
  ) : tensor<2x3xf32>

  return
}

func.func @reverse_dim1() {
  %input = util.unfoldable_constant dense<[[1, 2, 3],
                                           [4, 5, 6]]> : tensor<2x3xi32>

  %init = tensor.empty() : tensor<2x3xi32>
  %0 = iree_linalg_ext.reverse
         dimensions(dense<1> : tensor<1xi64>)
         ins(%input : tensor<2x3xi32>)
         outs(%init : tensor<2x3xi32>) : tensor<2x3xi32>

  check.expect_eq_const(
      %0,
      dense<[[3, 2, 1], [6, 5, 4]]> : tensor<2x3xi32>
  ) : tensor<2x3xi32>

  return
}

func.func @reverse_multi_dims() {
  %input = util.unfoldable_constant dense<[[1, 2, 3],
                                           [4, 5, 6]]> : tensor<2x3xi32>

  %init = tensor.empty() : tensor<2x3xi32>
  %0 = iree_linalg_ext.reverse
         dimensions(dense<[0, 1]> : tensor<2xi64>)
         ins(%input : tensor<2x3xi32>)
         outs(%init : tensor<2x3xi32>) : tensor<2x3xi32>

  check.expect_eq_const(
      %0,
      dense<[[6, 5, 4], [3, 2, 1]]> : tensor<2x3xi32>
  ) : tensor<2x3xi32>

  return
}
