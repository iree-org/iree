func.func @logical_right_shift() {
  %0 = util.unfoldable_constant dense<[5, 8, 9, 256]> : tensor<4xi32>
  %1 = util.unfoldable_constant dense<[0, 1, 2, 8]> : tensor<4xi32>
  %result = tosa.logical_right_shift %0, %1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[5, 4, 2, 1]> : tensor<4xi32>) : tensor<4xi32>
  return
}
