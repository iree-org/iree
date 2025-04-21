func.func @tensor_float() {
  %0 = util.unfoldable_constant dense<[1.0, 0.0, 4.5, 2.0]> : tensor<4xf32>
  %result = tosa.clamp %0 {min_val = 1.0 : f32, max_val = 4.0 : f32} : (tensor<4xf32>) -> tensor<4xf32>
  check.expect_almost_eq_const(%result, dense<[1.0, 1.0, 4.0, 2.0]> : tensor<4xf32>) : tensor<4xf32>
  return
}

// Tosa test failing, not sure why
// See #20422.
// func.func @tensor_int() {
//   %0 = util.unfoldable_constant dense<[1, 0, 5, 2]> : tensor<4xi32>
//   %result = tosa.clamp %0 {min_val = 1 : i32, max_val = 4 : i32} : (tensor<4xi32>) -> tensor<4xi32>
//   check.expect_eq_const(%result, dense<[1, 1, 4, 2]> : tensor<4xi32>) : tensor<4xi32>
//   return
// }
