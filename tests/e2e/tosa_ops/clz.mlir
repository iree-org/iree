func.func @tensor_int32() attributes { iree.module.export } {
  %0 = util.unfoldable_constant dense<[0, 1, -8, 256]> : tensor<4xi32>
  %result = tosa.clz %0 : (tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[32, 31, 0, 23]> : tensor<4xi32>) : tensor<4xi32>
  return
}
