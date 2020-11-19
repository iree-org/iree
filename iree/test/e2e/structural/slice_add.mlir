func @slice_whole_buffer() attributes { iree.module.export } {
  %input0 = iree.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]]> : tensor<3x4xi32>
  %input1 = iree.unfoldable_constant dense<10> : tensor<3x4xi32>
  %workload = constant 12 : index
  %result = flow.dispatch.region[%workload: index](%arg0 = %input0 : tensor<3x4xi32>, %arg1 = %input1 : tensor<3x4xi32>) -> tensor<3x4xi32> {
    %0 = "mhlo.slice"(%arg0) {
      start_indices = dense<[0, 0]> : tensor<2xi64>,
      limit_indices = dense<[3, 4]> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } : (tensor<3x4xi32>) -> tensor<3x4xi32>
    %1 = mhlo.add %0, %arg1 : tensor<3x4xi32>
    flow.return %1 : tensor<3x4xi32>
  }
  check.expect_eq_const(%result, dense<[
      [11, 12, 13, 14],
      [15, 16, 17, 18],
      [19, 20, 21, 22]]> : tensor<3x4xi32>) : tensor<3x4xi32>
  return
}

func @slice_whole_stride() attributes { iree.module.export } {
  %input0 = iree.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]]> : tensor<3x4xi32>
  %input1 = iree.unfoldable_constant dense<10> : tensor<1x4xi32>
  %workload = constant 4 : index
  %result = flow.dispatch.region[%workload: index](%arg0 = %input0 : tensor<3x4xi32>, %arg1 = %input1 : tensor<1x4xi32>) -> tensor<1x4xi32> {
    %0 = "mhlo.slice"(%arg0) {
      start_indices = dense<[1, 0]> : tensor<2xi64>,
      limit_indices = dense<[2, 4]> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } : (tensor<3x4xi32>) -> tensor<1x4xi32>
    %1 = mhlo.add %0, %arg1 : tensor<1x4xi32>
    flow.return %1 : tensor<1x4xi32>
  }
  check.expect_eq_const(%result, dense<[[15, 16, 17, 18]]> : tensor<1x4xi32>) : tensor<1x4xi32>
  return
}

func @slice_stride_part() attributes { iree.module.export } {
  %input0 = iree.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]]> : tensor<3x4xi32>
  %input1 = iree.unfoldable_constant dense<10> : tensor<1x2xi32>
  %workload = constant 2 : index
  %result = flow.dispatch.region[%workload: index](%arg0 = %input0 : tensor<3x4xi32>, %arg1 = %input1 : tensor<1x2xi32>) -> tensor<1x2xi32> {
    %0 = "mhlo.slice"(%arg0) {
      start_indices = dense<[1, 1]> : tensor<2xi64>,
      limit_indices = dense<[2, 3]> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } : (tensor<3x4xi32>) -> tensor<1x2xi32>
    %1 = mhlo.add %0, %arg1 : tensor<1x2xi32>
    flow.return %1 : tensor<1x2xi32>
  }
  check.expect_eq_const(%result, dense<[[16, 17]]> : tensor<1x2xi32>) : tensor<1x2xi32>
  return
}

func @slice_multi_stride() attributes { iree.module.export } {
  %input0 = iree.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]]> : tensor<3x4xi32>
  %input1 = iree.unfoldable_constant dense<10> : tensor<2x4xi32>
  %workload = constant 8 : index
  %result = flow.dispatch.region[%workload: index](%arg0 = %input0 : tensor<3x4xi32>, %arg1 = %input1 : tensor<2x4xi32>) -> tensor<2x4xi32> {
    %0 = "mhlo.slice"(%arg0) {
      start_indices = dense<[1, 0]> : tensor<2xi64>,
      limit_indices = dense<[3, 4]> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } : (tensor<3x4xi32>) -> tensor<2x4xi32>
    %1 = mhlo.add %0, %arg1 : tensor<2x4xi32>
    flow.return %1 : tensor<2x4xi32>
  }
  check.expect_eq_const(%result, dense<[
      [15, 16, 17, 18],
      [19, 20, 21, 22]]> : tensor<2x4xi32>) : tensor<2x4xi32>
  return
}
