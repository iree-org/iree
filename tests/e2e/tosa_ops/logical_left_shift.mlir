func.func @tensor() {
  %0 = util.unfoldable_constant dense<[5, 3, 2, 7]> : tensor<4xi32>
  %1 = util.unfoldable_constant dense<[0, 1, 2, 3]> : tensor<4xi32>
  %result = tosa.logical_left_shift %0, %1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[5, 6, 8, 56]> : tensor<4xi32>) : tensor<4xi32>
  return
}
