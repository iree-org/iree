// RUN: iree-run-mlir -iree-hal-target-backends=interpreter-bytecode %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @slice_whole_buffer
func @slice_whole_buffer() -> tensor<3x4xi32> {
  %input = iree.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]
  ]> : tensor<3x4xi32>
  %result = "xla_hlo.slice"(%input) {
    start_indices = dense<[0, 0]> : tensor<2xi64>,
    limit_indices = dense<[3, 4]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<3x4xi32>
  return %result : tensor<3x4xi32>
}
// CHECK: 3x4xi32=[1 2 3 4][5 6 7 8][9 10 11 12]

// CHECK-LABEL: EXEC @slice_whole_stride
func @slice_whole_stride() -> tensor<1x4xi32> {
  %input = iree.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]
  ]> : tensor<3x4xi32>
  %result = "xla_hlo.slice"(%input) {
    start_indices = dense<[1, 0]> : tensor<2xi64>,
    limit_indices = dense<[2, 4]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x4xi32>
  return %result : tensor<1x4xi32>
}
// CHECK: 1x4xi32=[5 6 7 8]

// CHECK-LABEL: EXEC @slice_stride_part
func @slice_stride_part() -> tensor<1x2xi32> {
  %input = iree.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]
  ]> : tensor<3x4xi32>
  %result = "xla_hlo.slice"(%input) {
    start_indices = dense<[1, 1]> : tensor<2xi64>,
    limit_indices = dense<[2, 3]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  return %result : tensor<1x2xi32>
}
// CHECK: 1x2xi32=[6 7]

// CHECK-LABEL: EXEC @slice_multi_stride
func @slice_multi_stride() -> tensor<2x4xi32> {
  %input = iree.unfoldable_constant dense<[
    [01, 02, 03, 04],
    [05, 06, 07, 08],
    [09, 10, 11, 12]
  ]> : tensor<3x4xi32>
  %result = "xla_hlo.slice"(%input) {
    start_indices = dense<[1, 0]> : tensor<2xi64>,
    limit_indices = dense<[3, 4]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<2x4xi32>
  return %result : tensor<2x4xi32>
}
// CHECK: 2x4xi32=[5 6 7 8][9 10 11 12]
