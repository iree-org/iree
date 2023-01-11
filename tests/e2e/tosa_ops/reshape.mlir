func.func @tensor_downrank() {
  %0 = util.unfoldable_constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %result = "tosa.reshape"(%0) { new_shape = array<i64: 4> } : (tensor<2x2xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[1, 2, 3, 4]> : tensor<4xi32>) : tensor<4xi32>
  return
}

func.func @tensor_uprank() {
  %0 = util.unfoldable_constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %result = "tosa.reshape"(%0) { new_shape = array<i64:2, 2> } : (tensor<4xi32>) -> tensor<2x2xi32>
  check.expect_eq_const(%result, dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

func.func @tensor_crossrank() {
  %0 = util.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %result = "tosa.reshape"(%0) { new_shape = array<i64:3, 2> } : (tensor<2x3xi32>) -> tensor<3x2xi32>
  check.expect_eq_const(%result, dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>) : tensor<3x2xi32>
  return
}

