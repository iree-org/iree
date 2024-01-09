func.func @transpose_2d() {
  %input = util.unfoldable_constant dense<[[1, 2, 3],
                                           [4, 5, 6]]> : tensor<2x3xi32>
  %0 = "stablehlo.transpose"(%input) {
    permutation = array<i64: 1, 0>
  } : (tensor<2x3xi32>) -> tensor<3x2xi32>
  check.expect_eq_const(%0, dense<[[1, 4],
                                   [2, 5],
                                   [3, 6]]> : tensor<3x2xi32>) : tensor<3x2xi32>
  return
}

func.func @transpose_3d() {
  %input = util.unfoldable_constant dense<[[[ 1,  2,  3],
                                            [ 4,  5,  6]],
                                           [[ 7,  8,  9],
                                            [10, 11, 12]]]> : tensor<2x2x3xi32>
  %0 = "stablehlo.transpose"(%input) {
    permutation = array<i64: 0, 2, 1>
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
