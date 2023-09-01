func.func @xla_concatenate() {
  %c0 = util.unfoldable_constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %c1 = util.unfoldable_constant dense<[[5, 6, 7], [8, 9, 10]]> : tensor<2x3xi32>
  %c2 = util.unfoldable_constant dense<[[11, 12], [13, 14]]> : tensor<2x2xi32>

  %0 = "stablehlo.concatenate"(%c0, %c1) {dimension = 1} : (tensor<2x2xi32>, tensor<2x3xi32>) -> tensor<2x5xi32>
  check.expect_eq_const(%0, dense<[[1, 2, 5, 6, 7], [3, 4, 8, 9, 10]]> : tensor<2x5xi32>) : tensor<2x5xi32>

  %1 = "stablehlo.concatenate"(%c1, %c0) {dimension = 1} : (tensor<2x3xi32>, tensor<2x2xi32>) -> tensor<2x5xi32>
  check.expect_eq_const(%1, dense<[[5, 6, 7, 1, 2], [8, 9, 10, 3, 4]]> : tensor<2x5xi32>) : tensor<2x5xi32>

  %2 = "stablehlo.concatenate"(%c0, %c1, %c2) {dimension = 1} : (tensor<2x2xi32>, tensor<2x3xi32>, tensor<2x2xi32>) -> tensor<2x7xi32>
  check.expect_eq_const(%2, dense<[[1, 2, 5, 6, 7, 11, 12], [3, 4, 8, 9, 10, 13, 14]]> : tensor<2x7xi32>) : tensor<2x7xi32>

  %3 = "stablehlo.concatenate"(%c0, %c2) {dimension = 0} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<4x2xi32>
  check.expect_eq_const(%3, dense<[[1, 2], [3, 4], [11, 12], [13, 14]]> : tensor<4x2xi32>) : tensor<4x2xi32>
  return
}

func.func @concatenate_cst() {
  %c0 = util.unfoldable_constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %c1 = stablehlo.constant dense<0> : tensor<2x3xi32>
  %0 = "stablehlo.concatenate"(%c0, %c1) {dimension = 1} : (tensor<2x2xi32>, tensor<2x3xi32>) -> tensor<2x5xi32>
  check.expect_eq_const(%0, dense<[[1, 2, 0, 0, 0], [3, 4, 0, 0, 0]]> : tensor<2x5xi32>) : tensor<2x5xi32>
  return
}
