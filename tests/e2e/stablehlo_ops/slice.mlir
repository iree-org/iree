func.func @slice_whole_buffer() {
  %input = util.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]]> : tensor<3x4xi32>
  %result = "stablehlo.slice"(%input) {
    start_indices = array<i64: 0, 0>,
    limit_indices = array<i64: 3, 4>,
    strides = array<i64: 1, 1>
  } : (tensor<3x4xi32>) -> tensor<3x4xi32>
  check.expect_eq_const(%result, dense<[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12]]> : tensor<3x4xi32>) : tensor<3x4xi32>
  return
}

func.func @slice_whole_stride() {
  %input = util.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]]> : tensor<3x4xi32>
  %result = "stablehlo.slice"(%input) {
    start_indices = array<i64: 1, 0>,
    limit_indices = array<i64: 2, 4>,
    strides = array<i64: 1, 1>
  } : (tensor<3x4xi32>) -> tensor<1x4xi32>
  check.expect_eq_const(%result, dense<[[5, 6, 7, 8]]> : tensor<1x4xi32>) : tensor<1x4xi32>
  return
}

func.func @slice_stride_part() {
  %input = util.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]]> : tensor<3x4xi32>
  %result = "stablehlo.slice"(%input) {
    start_indices = array<i64: 1, 1>,
    limit_indices = array<i64: 2, 3>,
    strides = array<i64: 1, 1>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  check.expect_eq_const(%result, dense<[[6, 7]]> : tensor<1x2xi32>) : tensor<1x2xi32>
  return
}

func.func @slice_multi_stride() {
  %input = util.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]]> : tensor<3x4xi32>
  %result = "stablehlo.slice"(%input) {
    start_indices = array<i64: 1, 0>,
    limit_indices = array<i64: 3, 4>,
    strides = array<i64: 1, 1>
  } : (tensor<3x4xi32>) -> tensor<2x4xi32>
  check.expect_eq_const(%result, dense<[
      [5, 6, 7, 8],
      [9, 10, 11, 12]]> : tensor<2x4xi32>) : tensor<2x4xi32>
  return
}
