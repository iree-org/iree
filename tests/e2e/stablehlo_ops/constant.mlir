func.func @high_rank () {
  %dense = stablehlo.constant dense<[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]> : tensor<2x2x3xi32>
  check.expect_eq_const(%dense, dense<[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]> : tensor<2x2x3xi32>) : tensor<2x2x3xi32>

  // %splat = stablehlo.constant dense<1> : tensor<2x2x3xi32>
  // check.expect_eq_const(%splat, dense<1> : tensor<2x2x3xi32>) : tensor<2x2x3xi32>
  return
}

// func.func @i8() {
//   %c = stablehlo.constant dense<[1, 2]> : tensor<2xi8>
//   check.expect_eq_const(%c, dense<[1, 2]> : tensor<2xi8>) : tensor<2xi8>
//   return
// }

// func.func @i32 () {
//   %c = stablehlo.constant dense<[1, 2]> : tensor<2xi32>
//   check.expect_eq_const(%c,  dense<[1, 2]> : tensor<2xi32>) : tensor<2xi32>
//   return
// }

// func.func @f32 () {
//   %c = stablehlo.constant dense<[1.1, 2.1]> : tensor<2xf32>
//   check.expect_almost_eq_const(%c, dense<[1.1, 2.1]> : tensor<2xf32>) : tensor<2xf32>
//   return
// }
