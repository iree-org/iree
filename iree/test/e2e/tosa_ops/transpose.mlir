func @test_transpose() attributes { iree.module.export } {
  %0 = constant dense<[[[0, 1, 2], [3, 4, 5]]]> : tensor<1x2x3xi32>
  %1 = constant dense<[1, 2, 0]> : tensor<3xi32>
  %2 = "tosa.transpose"(%0, %1) : (tensor<1x2x3xi32>, tensor<3xi32>) -> (tensor<2x3x1xi32>)
  check.expect_eq_const(%2, dense<[[[0], [1], [2]], [[3], [4], [5]]]> : tensor<2x3x1xi32>) : tensor<2x3x1xi32>
  return
}
