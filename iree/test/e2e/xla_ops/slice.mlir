func @slice_whole_buffer() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]]> : tensor<3x4xi32>
  %start_indices = constant dense<[0, 0]> : tensor<2xi64>
  %limit_indices = constant dense<[3, 4]> : tensor<2xi64>
  %strides = constant dense<1> : tensor<2xi64>
  %result = "mhlo.slice"(%input, %start_indices, %limit_indices, %strides)
    : (tensor<3x4xi32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<3x4xi32>
  check.expect_eq_const(%result, dense<[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12]]> : tensor<3x4xi32>) : tensor<3x4xi32>
  return
}

func @slice_whole_stride() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]]> : tensor<3x4xi32>
  %start_indices = constant dense<[1, 0]> : tensor<2xi64>
  %limit_indices = constant dense<[2, 4]> : tensor<2xi64>
  %strides = constant dense<1> : tensor<2xi64>
  %result = "mhlo.slice"(%input, %start_indices, %limit_indices, %strides)
    : (tensor<3x4xi32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x4xi32>
  check.expect_eq_const(%result, dense<[[5, 6, 7, 8]]> : tensor<1x4xi32>) : tensor<1x4xi32>
  return
}

func @slice_stride_part() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]]> : tensor<3x4xi32>
  %start_indices = constant dense<[1, 1]> : tensor<2xi64>
  %limit_indices = constant dense<[2, 3]> : tensor<2xi64>
  %strides = constant dense<1> : tensor<2xi64>
  %result = "mhlo.slice"(%input, %start_indices, %limit_indices, %strides)
    : (tensor<3x4xi32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<1x2xi32>
  check.expect_eq_const(%result, dense<[[6, 7]]> : tensor<1x2xi32>) : tensor<1x2xi32>
  return
}

func @slice_multi_stride() attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]]> : tensor<3x4xi32>
  %start_indices = constant dense<[1, 0]> : tensor<2xi64>
  %limit_indices = constant dense<[3, 4]> : tensor<2xi64>
  %strides = constant dense<1> : tensor<2xi64>
  %result = "mhlo.slice"(%input, %start_indices, %limit_indices, %strides)
    : (tensor<3x4xi32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<2x4xi32>
  check.expect_eq_const(%result, dense<[
      [5, 6, 7, 8],
      [9, 10, 11, 12]]> : tensor<2x4xi32>) : tensor<2x4xi32>
  return
}
