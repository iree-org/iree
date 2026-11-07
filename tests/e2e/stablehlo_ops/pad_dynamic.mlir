// pad of dynamically shaped operands with negative edge padding.

func.func @pad_negative_and_interior() {
  %a = flow.tensor.dynamic_constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]> : tensor<3x4xf32> -> tensor<?x?xf32>
  %v = stablehlo.constant dense<-1.0> : tensor<f32>
  %r = stablehlo.pad %a, %v, low = [1, -1], high = [-2, 3], interior = [0, 1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  %s = tensor.cast %r : tensor<?x?xf32> to tensor<2x9xf32>
  check.expect_eq_const(%s, dense<[[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, 2.0, -1.0, 3.0, -1.0, 4.0, -1.0, -1.0, -1.0]]> : tensor<2x9xf32>) : tensor<2x9xf32>
  return
}

func.func @pad_negative_low_and_high() {
  %a = flow.tensor.dynamic_constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]> : tensor<3x4xf32> -> tensor<?x?xf32>
  %v = stablehlo.constant dense<-1.0> : tensor<f32>
  %r = stablehlo.pad %a, %v, low = [-1, 2], high = [1, -2], interior = [0, 0] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  %s = tensor.cast %r : tensor<?x?xf32> to tensor<3x4xf32>
  check.expect_eq_const(%s, dense<[[-1.0, -1.0, 5.0, 6.0], [-1.0, -1.0, 9.0, 10.0], [-1.0, -1.0, -1.0, -1.0]]> : tensor<3x4xf32>) : tensor<3x4xf32>
  return
}

// The dilated input does not fit i64, but its cropped result has three elements.
func.func @dynamic_pad_extreme_crop() {
  %input = util.unfoldable_constant dense<[1, 2, 3]> : tensor<3xi32>
  %pad = stablehlo.constant dense<7> : tensor<i32>
  %lo = util.unfoldable_constant dense<-9223372036854775804> : tensor<1xi64>
  %hi = util.unfoldable_constant dense<-9223372036854775806> : tensor<1xi64>
  %in = util.unfoldable_constant dense<9223372036854775805> : tensor<1xi64>
  %r = stablehlo.dynamic_pad %input, %pad, %lo, %hi, %in : (tensor<3xi32>, tensor<i32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi32>
  %size = stablehlo.get_dimension_size %r, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  check.expect_eq_const(%size, dense<3> : tensor<i32>) : tensor<i32>
  %fixed = tensor.cast %r : tensor<?xi32> to tensor<3xi32>
  check.expect_eq_const(%fixed, dense<[7, 7, 2]> : tensor<3xi32>) : tensor<3xi32>
  return
}

// INT64_MIN cannot be negated in signed64 arithmetic. Its crop offset is
// nevertheless representable as an unsigned coordinate into the virtual pad.
func.func @dynamic_pad_minimum_low() {
  %input = util.unfoldable_constant dense<[11, 22]> : tensor<2xi32>
  %pad = stablehlo.constant dense<7> : tensor<i32>
  %lo = util.unfoldable_constant dense<-9223372036854775808> : tensor<1xi64>
  %hi = util.unfoldable_constant dense<3> : tensor<1xi64>
  %in = util.unfoldable_constant dense<9223372036854775806> : tensor<1xi64>
  %r = stablehlo.dynamic_pad %input, %pad, %lo, %hi, %in : (tensor<2xi32>, tensor<i32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi32>
  check.expect_eq_const(%r, dense<7> : tensor<3xi32>) : tensor<3xi32>
  return
}

// The singleton has no gaps even when the logical unsigned stride is 2^64.
func.func @dynamic_pad_unsigned_maximum_interior() {
  %input = util.unfoldable_constant dense<42> : tensor<1xi32>
  %dynamic = flow.tensor.dynamic_constant dense<42> : tensor<1xi32> -> tensor<?xi32>
  %pad = stablehlo.constant dense<7> : tensor<i32>
  %zero_bits = util.unfoldable_constant dense<0> : tensor<1xi64>
  %max_bits = util.unfoldable_constant dense<-1> : tensor<1xi64>
  %zero = stablehlo.convert %zero_bits : (tensor<1xi64>) -> tensor<1xui64>
  %max = stablehlo.convert %max_bits : (tensor<1xi64>) -> tensor<1xui64>
  %r = stablehlo.dynamic_pad %dynamic, %pad, %zero, %zero, %max : (tensor<?xi32>, tensor<i32>, tensor<1xui64>, tensor<1xui64>, tensor<1xui64>) -> tensor<?xi32>
  %size = stablehlo.get_dimension_size %r, dim = 0 : (tensor<?xi32>) -> tensor<i32>
  check.expect_eq_const(%size, dense<1> : tensor<i32>) : tensor<i32>
  %fixed = tensor.cast %r : tensor<?xi32> to tensor<1xi32>
  check.expect_eq(%fixed, %input) : tensor<1xi32>
  return
}
