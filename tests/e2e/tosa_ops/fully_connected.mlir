func.func @tensor_float() {
  %0 = util.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %1 = util.unfoldable_constant dense<[[7.0, 8.0, 9.0]]> : tensor<1x3xf32>
  %2 = util.unfoldable_constant dense<[1.0]> : tensor<1xf32>
  %result = "tosa.fully_connected"(%0, %1, %2) : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<1xf32>) -> tensor<2x1xf32>
  check.expect_eq_const(%result, dense<[[51.0], [123.0]]> : tensor<2x1xf32>) : tensor<2x1xf32>
  return
}
