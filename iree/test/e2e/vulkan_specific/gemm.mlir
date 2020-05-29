// RUN: iree-run-mlir -iree-hal-target-backends=vulkan-spirv -iree-spirv-workgroup-size=2,2 %s

func @gemm() -> tensor<4x4xi32> {
  %0 = iree.unfoldable_constant dense<[[1, 2, 3, 4],
                                       [5, 6, 7, 8],
                                       [9, 10, 11, 12],
                                       [13, 14, 15, 16]]> : tensor<4x4xi32>
  %1 = iree.unfoldable_constant dense<[[1, 2, 3, 4],
                                       [5, 6, 7, 8],
                                       [9, 10, 11, 12],
                                       [13, 14, 15, 16]]> : tensor<4x4xi32>
  %2 = "xla_hlo.dot"(%0, %1) { precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  return %2 : tensor<4x4xi32>
}
//      CHECK: 4x4xi32=
// CHECK-SAME: [
// CHECK-SAME:  [ 90, 100, 110, 120],
// CHECK-SAME:  [202, 228, 254, 280],
// CHECK-SAME:  [314, 356, 398, 440],
// CHECK-SAME:  [426, 484, 542, 600]
// CHECK-SAME: ]
