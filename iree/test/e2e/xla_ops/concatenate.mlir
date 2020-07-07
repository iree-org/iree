func @xla_concatenate() attributes { iree.module.export } {
  %c0 = iree.unfoldable_constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %c1 = iree.unfoldable_constant dense<[[5, 6, 7], [8, 9, 10]]> : tensor<2x3xi32>
  %c2 = iree.unfoldable_constant dense<[[11, 12], [13, 14]]> : tensor<2x2xi32>

  %0 = "mhlo.concatenate"(%c0, %c1) {dimension = 1} : (tensor<2x2xi32>, tensor<2x3xi32>) -> tensor<2x5xi32>
  check.expect_eq_const(%0, dense<[[1, 2, 5, 6, 7], [3, 4, 8, 9, 10]]> : tensor<2x5xi32>) : tensor<2x5xi32>

  %1 = "mhlo.concatenate"(%c1, %c0) {dimension = 1} : (tensor<2x3xi32>, tensor<2x2xi32>) -> tensor<2x5xi32>
  check.expect_eq_const(%1, dense<[[5, 6, 7, 1, 2], [8, 9, 10, 3, 4]]> : tensor<2x5xi32>) : tensor<2x5xi32>

  %2 = "mhlo.concatenate"(%c0, %c1, %c2) {dimension = 1} : (tensor<2x2xi32>, tensor<2x3xi32>, tensor<2x2xi32>) -> tensor<2x7xi32>
  check.expect_eq_const(%2, dense<[[1, 2, 5, 6, 7, 11, 12], [3, 4, 8, 9, 10, 13, 14]]> : tensor<2x7xi32>) : tensor<2x7xi32>

  %3 = "mhlo.concatenate"(%c0, %c2) {dimension = 0} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<4x2xi32>
  check.expect_eq_const(%3, dense<[[1, 2], [3, 4], [11, 12], [13, 14]]> : tensor<4x2xi32>) : tensor<4x2xi32>
  return
}
