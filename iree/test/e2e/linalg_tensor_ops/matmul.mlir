func @tensor() attributes { iree.module.export } {
  %A = iree.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %B = iree.unfoldable_constant dense<[[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]]> : tensor<3x4xf32>
  %C = iree.unfoldable_constant dense<1000.0> : tensor<2x4xf32>

  %D = linalg.matmul ins(%A, %B: tensor<2x3xf32>, tensor<3x4xf32>)
                    outs(%C: tensor<2x4xf32>) -> tensor<2x4xf32>

  check.expect_almost_eq_const(%D, dense<[[1038.0, 1044.0, 1050.0, 1056.0],
                                          [1083.0, 1098.0, 1113.0, 1128.0]]> :
    tensor<2x4xf32>) : tensor<2x4xf32>

  return
}


func @large() attributes { iree.module.export } {
  %lhs = iree.unfoldable_constant dense<1.0> : tensor<250x1024xf32>
  %rhs = iree.unfoldable_constant dense<0.4> : tensor<1024x500xf32>
  %init = iree.unfoldable_constant dense<0.2> : tensor<250x500xf32>
  %res = linalg.matmul
    ins(%lhs, %rhs : tensor<250x1024xf32>, tensor<1024x500xf32>)
    outs(%init : tensor<250x500xf32>) -> tensor<250x500xf32>
  check.expect_almost_eq_const(%res, dense<409.796> : tensor<250x500xf32>) : tensor<250x500xf32>
  return
}
