func.func @expand_shape_i32() {
  %1 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %2 = util.optimization_barrier %1 : tensor<4xi32>
  %expanded = tensor.expand_shape %2 [[0, 1]] output_shape [1, 4] : tensor<4xi32> into tensor<1x4xi32>
  check.expect_eq_const(%expanded, dense<[[1,2,3,4]]> : tensor<1x4xi32>) : tensor<1x4xi32>
  return
}

func.func @expand_shape_i64() {
  %1 = arith.constant dense<[1, 2, 3, 4]> : tensor<4xi64>
  %2 = util.optimization_barrier %1 : tensor<4xi64>
  %expanded = tensor.expand_shape %2 [[0, 1]] output_shape [1, 4] : tensor<4xi64> into tensor<1x4xi64>
  check.expect_eq_const(%expanded, dense<[[1,2,3,4]]> : tensor<1x4xi64>) : tensor<1x4xi64>
  return
}
