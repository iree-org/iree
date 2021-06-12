func @tensor_float() {
  %0 = iree.unfoldable_constant dense<[1.0, -1.0, 3.0, 5.0]> : tensor<4xf32>
  %result = "tosa.reluN"(%0) {max_fp = 4.0 : f32, max_int = 4 : i64} : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[1.0, 0.0, 3.0, 4.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func @tensor_int() {
  %0 = iree.unfoldable_constant dense<[1, -1, 3, 5]> : tensor<4xi32>
  %result = "tosa.reluN"(%0) {max_fp = 4.0 : f32, max_int = 4 : i64} : (tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[1, 0, 3, 4]> : tensor<4xi32>) : tensor<4xi32>
  return
}
