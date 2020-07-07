func @transpose_2d() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[[1, 2, 3],
                                           [4, 5, 6]]> : tensor<2x3xi32>
  %0 = "mhlo.transpose"(%input) {
    permutation = dense<[1, 0]> : tensor<2xi64>
  } : (tensor<2x3xi32>) -> tensor<3x2xi32>
  check.expect_eq_const(%0, dense<[[1, 4],
                                   [2, 5],
                                   [3, 6]]> : tensor<3x2xi32>) : tensor<3x2xi32>
  return
}

func @transpose_3d() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[[[ 1,  2,  3],
                                            [ 4,  5,  6]],
                                           [[ 7,  8,  9],
                                            [10, 11, 12]]]> : tensor<2x2x3xi32>
  %0 = "mhlo.transpose"(%input) {
    permutation = dense<[0, 2, 1]> : tensor<3xi64>
  } : (tensor<2x2x3xi32>) -> tensor<2x3x2xi32>
  check.expect_eq_const(%0, dense<[
    [[ 1,  4],
     [ 2,  5],
     [ 3,  6]],
    [[ 7, 10],
     [ 8, 11],
     [ 9, 12]]]> : tensor<2x3x2xi32>) : tensor<2x3x2xi32>
  return
}
