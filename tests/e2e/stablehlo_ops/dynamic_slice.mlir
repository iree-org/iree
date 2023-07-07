func.func @dynamic_slice() {
  %input = util.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]]> : tensor<3x4xi32>
  %start1 = util.unfoldable_constant dense<1> : tensor<i64>
  %start2 = util.unfoldable_constant dense<2> : tensor<i64>
  %result = "stablehlo.dynamic_slice"(%input, %start1, %start2) {
    slice_sizes = dense<[2, 2]> : tensor<2xi64>
  } : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x2xi32>
  check.expect_eq_const(%result, dense<[
      [7, 8],
      [11, 12]]> : tensor<2x2xi32>) : tensor<2x2xi32>
  return
}

func.func @dynamic_unit_slice() {
  %input = util.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]]> : tensor<3x4xi32>
  %start1 = util.unfoldable_constant dense<1> : tensor<i64>
  %start2 = util.unfoldable_constant dense<2> : tensor<i64>
  %result = "stablehlo.dynamic_slice"(%input, %start1, %start2) {
    slice_sizes = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x2xi32>
  check.expect_eq_const(%result, dense<[
      [7, 8]]> : tensor<1x2xi32>) : tensor<1x2xi32>
  return
}

func.func @dynamic_1d_slice() {
  %input = util.unfoldable_constant dense<[1, 2, 3, 4]> : tensor<4xi32>
  %start1 = util.unfoldable_constant dense<1> : tensor<i64>
  %result = "stablehlo.dynamic_slice"(%input, %start1) {
    slice_sizes = dense<[2]> : tensor<1xi64>
  } : (tensor<4xi32>, tensor<i64>) -> tensor<2xi32>
  check.expect_eq_const(%result, dense<[2, 3]> : tensor<2xi32>) : tensor<2xi32>
  return
}
