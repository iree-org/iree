func @tensor_i8() {
  %0 = util.unfoldable_constant dense<[[[[1], [2], [3], [4]], [[5], [6], [7], [8]]]]> : tensor<1x2x4x1xi8>
  %result = "tosa.max_pool2d"(%0) {kernel = [2, 2], stride = [1, 1], pad = [0, 0, 0, 0]} : (tensor<1x2x4x1xi8>) -> tensor<1x1x3x1xi8>
  check.expect_eq_const(%result, dense<[[[[6], [7], [8]]]]> : tensor<1x1x3x1xi8>) : tensor<1x1x3x1xi8>
  return
}

func @tensor_i16() {
  %0 = util.unfoldable_constant dense<[[[[1], [2], [3], [4]], [[5], [6], [7], [8]]]]> : tensor<1x2x4x1xi16>
  %result = "tosa.max_pool2d"(%0) {kernel = [2, 2], stride = [1, 1], pad = [0, 0, 0, 0]} : (tensor<1x2x4x1xi16>) -> tensor<1x1x3x1xi16>
  check.expect_eq_const(%result, dense<[[[[6], [7], [8]]]]> : tensor<1x1x3x1xi16>) : tensor<1x1x3x1xi16>
  return
}

func @tensor_i32() {
  %0 = util.unfoldable_constant dense<[[[[1], [2], [3], [4]], [[5], [6], [7], [8]]]]> : tensor<1x2x4x1xi32>
  %result = "tosa.max_pool2d"(%0) {kernel = [2, 2], stride = [1, 1], pad = [0, 0, 0, 0]} : (tensor<1x2x4x1xi32>) -> tensor<1x1x3x1xi32>
  check.expect_eq_const(%result, dense<[[[[6], [7], [8]]]]> : tensor<1x1x3x1xi32>) : tensor<1x1x3x1xi32>
  return
}

func @tensor_f32() {
  %0 = util.unfoldable_constant dense<[[[[1.], [2.], [3.], [4.]], [[5.], [6.], [7.], [8.]]]]> : tensor<1x2x4x1xf32>
  %result = "tosa.max_pool2d"(%0) {kernel = [2, 2], stride = [1, 1], pad = [0, 0, 0, 0]} : (tensor<1x2x4x1xf32>) -> tensor<1x1x3x1xf32>
  check.expect_eq_const(%result, dense<[[[[6.], [7.], [8.]]]]> : tensor<1x1x3x1xf32>) : tensor<1x1x3x1xf32>
  return
}

