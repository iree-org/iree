func.func @tensor_float() {
  %0 = util.unfoldable_constant dense<[1.0, 0.0, 4.5, 2.0]> : tensor<4xf32>
  %result = "tosa.clamp"(%0) {min_int = 1 : i64, max_int = 4 : i64, min_fp = 1.0 : f32, max_fp = 4.0 : f32} : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[1.0, 1.0, 4.0, 2.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

func.func @tensor_int() {
  %0 = util.unfoldable_constant dense<[1, 0, 5, 2]> : tensor<4xi32>
  %result = "tosa.clamp"(%0) {min_int = 1 : i64, max_int = 4 : i64, min_fp = 1.0 : f32, max_fp = 4.0 : f32} : (tensor<4xi32>) -> tensor<4xi32>
  check.expect_eq_const(%result, dense<[1, 1, 4, 2]> : tensor<4xi32>) : tensor<4xi32>
  return
}
