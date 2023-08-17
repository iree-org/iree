func.func @tensor() {
  %0 = util.unfoldable_constant dense<[0x0, 0x011, 0x101, 0x111]> : tensor<4xi32>
  %1 = util.unfoldable_constant dense<[0x0, 0x010, 0x111, 0x000]> : tensor<4xi32>
  %result = tosa.bitwise_and %0, %1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[0x0, 0x010, 0x101, 0x000]> : tensor<4xi32>) : tensor<4xi32>
  return
}
