func @tensor_downrank() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %result = "tosa.reshape"(%0) { new_shape = [4] } : (tensor<2x2xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[1, 2, 3, 4]> : tensor<4xi32>) : tensor<4xi32>
  return
}

func @tensor_uprank() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %result = "tosa.reshape"(%0) { new_shape = [2, 2] } : (tensor<4xi32>) -> tensor<2x2xi32>
  check.expect_eq_const(%result, dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

func @tensor_crossrank() attributes { iree.module.export } {
  %0 = iree.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %result = "tosa.reshape"(%0) { new_shape = [3, 2] } : (tensor<2x3xi32>) -> tensor<3x2xi32>
  check.expect_eq_const(%result, dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>) : tensor<3x2xi32>
  return
}

