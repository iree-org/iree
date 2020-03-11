// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @broadcast_2D_3D
func @broadcast_2D_3D() -> tensor<3x2x4xi32> {
  %input = iree.unfoldable_constant dense<[[1, 2, 3, 4],
                                           [5, 6, 7, 8]]> : tensor<2x4xi32>
  %0 = "xla_hlo.broadcast"(%input) {broadcast_sizes = dense<[3]> : tensor<1xi64>} : (tensor<2x4xi32>) -> tensor<3x2x4xi32>
  return %0 : tensor<3x2x4xi32>
}
// CHECK: 3x2x4xi32=[
// CHECK-SAME: [1 2 3 4]
// CHECK-SAME: [5 6 7 8]
// CHECK-SAME: ][
// CHECK-SAME: [1 2 3 4]
// CHECK-SAME: [5 6 7 8]
// CHECK-SAME: ][
// CHECK-SAME: [1 2 3 4]
// CHECK-SAME: [5 6 7 8]
// CHECK-SAME: ]

// -----

// CHECK-LABEL: EXEC @broadcast_in_dim_2D_3D
func @broadcast_in_dim_2D_3D() -> tensor<3x2x4xi32> {
  %input = iree.unfoldable_constant dense<[[1, 2, 3, 4],
                                           [5, 6, 7, 8]]> : tensor<2x4xi32>
  %0 = "xla_hlo.broadcast_in_dim"(%input) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xi32>) -> tensor<3x2x4xi32>
  return %0 : tensor<3x2x4xi32>
}
// CHECK: 3x2x4xi32=[
// CHECK-SAME: [1 2 3 4]
// CHECK-SAME: [5 6 7 8]
// CHECK-SAME: ][
// CHECK-SAME: [1 2 3 4]
// CHECK-SAME: [5 6 7 8]
// CHECK-SAME: ][
// CHECK-SAME: [1 2 3 4]
// CHECK-SAME: [5 6 7 8]
// CHECK-SAME: ]

// -----

// CHECK-LABEL: EXEC @broadcast_3D_scalar
func @broadcast_3D_scalar() -> tensor<3x2x4xi32> {
  %input = iree.unfoldable_constant dense<42> : tensor<i32>
  %0 = "xla_hlo.broadcast"(%input) {broadcast_sizes = dense<[3, 2, 4]> : tensor<3xi64>} : (tensor<i32>) -> tensor<3x2x4xi32>
  return %0 : tensor<3x2x4xi32>
}
// CHECK: 3x2x4xi32=[
// CHECK-SAME: [42 42 42 42]
// CHECK-SAME: [42 42 42 42]
// CHECK-SAME: ][
// CHECK-SAME: [42 42 42 42]
// CHECK-SAME: [42 42 42 42]
// CHECK-SAME: ][
// CHECK-SAME: [42 42 42 42]
// CHECK-SAME: [42 42 42 42]
// CHECK-SAME: ]

// -----

// CHECK-LABEL: EXEC @broadcast_in_dim_3D_scalar
func @broadcast_in_dim_3D_scalar() -> tensor<3x2x4xi32> {
  %input = iree.unfoldable_constant dense<42> : tensor<i32>
  %0 = "xla_hlo.broadcast_in_dim"(%input) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<i32>) -> tensor<3x2x4xi32>
  return %0 : tensor<3x2x4xi32>
}
// CHECK: 3x2x4xi32=[
// CHECK-SAME: [42 42 42 42]
// CHECK-SAME: [42 42 42 42]
// CHECK-SAME: ][
// CHECK-SAME: [42 42 42 42]
// CHECK-SAME: [42 42 42 42]
// CHECK-SAME: ][
// CHECK-SAME: [42 42 42 42]
// CHECK-SAME: [42 42 42 42]
// CHECK-SAME: ]
