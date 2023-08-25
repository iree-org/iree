func.func @tensor() {
  %0 = util.unfoldable_constant dense<[0x0, 0x11, 0x1101, 0x111]> : tensor<4xi32>
  %1 = util.unfoldable_constant dense<[0x0, 0x10, 0x0111, 0x111]> : tensor<4xi32>
  %result = tosa.bitwise_or %0, %1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[0x0, 0x11, 0x1111, 0x111]> : tensor<4xi32>) : tensor<4xi32>
  return
}
