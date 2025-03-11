func.func @tensor_float() {
  %0 = util.unfoldable_constant dense<[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]> : tensor<2x2x3xf32>
  %1 = util.unfoldable_constant dense<[[[7.0], [8.0], [9.0]], [[7.0], [8.0], [9.0]]]> : tensor<2x3x1xf32>
  %azp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %bzp0 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %result = tosa.matmul %0, %1, %azp0, %bzp0 : (tensor<2x2x3xf32>, tensor<2x3x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x2x1xf32>
  check.expect_eq_const(%result, dense<[[[50.0], [122.0]], [[50.0], [122.0]]]> : tensor<2x2x1xf32>) : tensor<2x2x1xf32>
  return
}

func.func @tensor_int() {
  %0 = util.unfoldable_constant dense<[[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]> : tensor<2x2x3xi32>
  %1 = util.unfoldable_constant dense<[[[7], [8], [9]], [[7], [8], [9]]]> : tensor<2x3x1xi32>
  %azp0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %bzp0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %result = tosa.matmul %0, %1, %azp0, %bzp0 : (tensor<2x2x3xi32>, tensor<2x3x1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2x2x1xi32>
  check.expect_eq_const(%result, dense<[[[50], [122]], [[50], [122]]]> : tensor<2x2x1xi32>) : tensor<2x2x1xi32>
  return
}
