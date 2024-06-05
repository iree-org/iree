func.func @tensor_float() {
  %0 = util.unfoldable_constant dense<[1.0, -1.5, 7.0, -2.0]> : tensor<4xf32>
  %1 = util.unfoldable_constant dense<[5.0, 1.0, 7.0, -3.0]> : tensor<4xf32>
  %result = tosa.greater %0, %1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  check.expect_eq_const(%result, dense<[false, false, false, true]> : tensor<4xi1>) : tensor<4xi1>
  return
}

func.func @tensor_int() {
  %0 = util.unfoldable_constant dense<[1, 0, 5, 3]> : tensor<4xi32>
  %1 = util.unfoldable_constant dense<[5, 0, 1, 8]> : tensor<4xi32>
  %result = tosa.greater %0, %1 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  check.expect_eq_const(%result, dense<[false, false, true, false]> : tensor<4xi1>) : tensor<4xi1>
  return
}
