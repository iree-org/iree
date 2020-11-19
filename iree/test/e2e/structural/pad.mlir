func @slice_add_pad_cst() attributes { iree.module.export } {
  %input0 = iree.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]]> : tensor<3x4xi32>
  %input1 = iree.unfoldable_constant dense<10> : tensor<2x4xi32>
  %workload = constant 8 : index
  %result = flow.dispatch.region[%workload: index](%arg0 = %input0 : tensor<3x4xi32>, %arg1 = %input1 : tensor<2x4xi32>) -> tensor<5x8xi32> {
    %0 = "mhlo.slice"(%arg0) {
      start_indices = dense<[1, 0]> : tensor<2xi64>,
      limit_indices = dense<[3, 4]> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } : (tensor<3x4xi32>) -> tensor<2x4xi32>
    %1 = mhlo.add %0, %arg1 : tensor<2x4xi32>
    %cst = constant dense<0> : tensor<i32>
    %2 = "mhlo.pad"(%1, %cst) {
      edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
      edge_padding_low = dense<[1, 1]> : tensor<2xi64>,
      interior_padding = dense<0> : tensor<2xi64>
    } : (tensor<2x4xi32>, tensor<i32>) -> tensor<5x8xi32>
    flow.return %2 : tensor<5x8xi32>
  }
  check.expect_eq_const(%result, dense<[
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 15, 16, 17, 18, 0, 0, 0],
    [0, 19, 20, 21, 22, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]]> : tensor<5x8xi32>) : tensor<5x8xi32>
  return
}

func @slice_add_pad_memref() attributes { iree.module.export } {
  %input0 = iree.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]]> : tensor<3x4xi32>
  %input1 = iree.unfoldable_constant dense<10> : tensor<2x4xi32>
  %input2 = iree.unfoldable_constant dense<42> : tensor<i32>
  %workload = constant 8 : index
  %result = flow.dispatch.region[%workload: index](%arg0 = %input0 : tensor<3x4xi32>, %arg1 = %input1 : tensor<2x4xi32>, %arg2 = %input2 : tensor<i32>) -> tensor<5x8xi32> {
    %0 = "mhlo.slice"(%arg0) {
      start_indices = dense<[1, 0]> : tensor<2xi64>,
      limit_indices = dense<[3, 4]> : tensor<2xi64>,
      strides = dense<1> : tensor<2xi64>
    } : (tensor<3x4xi32>) -> tensor<2x4xi32>
    %1 = mhlo.add %0, %arg1 : tensor<2x4xi32>
    %2 = "mhlo.pad"(%1, %arg2) {
      edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
      edge_padding_low = dense<[1, 1]> : tensor<2xi64>,
      interior_padding = dense<0> : tensor<2xi64>
    } : (tensor<2x4xi32>, tensor<i32>) -> tensor<5x8xi32>
    flow.return %2 : tensor<5x8xi32>
  }
  check.expect_eq_const(%result, dense<[
    [42, 42, 42, 42, 42, 42, 42, 42],
    [42, 15, 16, 17, 18, 42, 42, 42],
    [42, 19, 20, 21, 22, 42, 42, 42],
    [42, 42, 42, 42, 42, 42, 42, 42],
    [42, 42, 42, 42, 42, 42, 42, 42]]> : tensor<5x8xi32>) : tensor<5x8xi32>
  return
}


