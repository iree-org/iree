func.func @reduce_max_float() {
  %0 = util.unfoldable_constant dense<[[2.0, 2.0], [3.0, 4.0], [1.0, 0.0]]> : tensor<3x2xf32>
  %result = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<3x2xf32>) -> tensor<1x2xf32>
  check.expect_almost_eq_const(%result, dense<[[3.0, 4.0]]> : tensor<1x2xf32>) : tensor<1x2xf32>
  return
}

func.func @reduce_max_int() {
  %0 = util.unfoldable_constant dense<[[2, 2], [3, 4], [1, 0]]> : tensor<3x2xi32>
  %result = tosa.reduce_max %0 {axis = 0 : i32} : (tensor<3x2xi32>) -> tensor<1x2xi32>
  check.expect_eq_const(%result, dense<[[3, 4]]> : tensor<1x2xi32>) : tensor<1x2xi32>
  return
}

func.func @reduce_min_float() {
  %0 = util.unfoldable_constant dense<[[2.0, 2.0], [3.0, 4.0], [1.0, 0.0]]> : tensor<3x2xf32>
  %result = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<3x2xf32>) -> tensor<1x2xf32>
  check.expect_almost_eq_const(%result, dense<[[1.0, 0.0]]> : tensor<1x2xf32>) : tensor<1x2xf32>
  return
}

func.func @reduce_min_int() {
  %0 = util.unfoldable_constant dense<[[2, 2], [3, 4], [1, 0]]> : tensor<3x2xi32>
  %result = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<3x2xi32>) -> tensor<1x2xi32>
  check.expect_eq_const(%result, dense<[[1, 0]]> : tensor<1x2xi32>) : tensor<1x2xi32>
  return
}

func.func @reduce_prod_float() {
  %0 = util.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %result = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<2x3xf32>) -> tensor<1x3xf32>
  check.expect_almost_eq_const(%result, dense<[[4.0, 10.0, 18.0]]> : tensor<1x3xf32>) : tensor<1x3xf32>
  return
}

// Temporary disable because Tosa doesn't like these anymore, probably a bug there
// See #20422.
// func.func @reduce_prod_int() {
//   %0 = util.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
//   %result = tosa.reduce_product %0 {axis = 0 : i32} : (tensor<2x3xi32>) -> tensor<1x3xi32>
//   check.expect_eq_const(%result, dense<[[4, 10, 18]]> : tensor<1x3xi32>) : tensor<1x3xi32>
//   return
// }

func.func @reduce_sum_float() {
  %0 = util.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %result = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<2x3xf32>) -> tensor<1x3xf32>
  check.expect_almost_eq_const(%result, dense<[[5.0, 7.0, 9.0]]> : tensor<1x3xf32>) : tensor<1x3xf32>
  return
}

func.func @reduce_sum_int() {
  %0 = util.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %result = tosa.reduce_sum %0 {axis = 0 : i32} : (tensor<2x3xi32>) -> tensor<1x3xi32>
  check.expect_eq_const(%result, dense<[[5, 7, 9]]> : tensor<1x3xi32>) : tensor<1x3xi32>
  return
}

func.func @reduce_sum_float_axis_1() {
  %0 = util.unfoldable_constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %result = tosa.reduce_sum %0 {axis = 1 : i32} : (tensor<2x3xf32>) -> tensor<2x1xf32>
  check.expect_almost_eq_const(%result, dense<[[6.0], [15.0]]> : tensor<2x1xf32>) : tensor<2x1xf32>
  return
}
