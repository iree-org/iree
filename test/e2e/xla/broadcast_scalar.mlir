// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv -input-value="i32= 42" %s | IreeFileCheck %s)

// CHECK-LABEL: @broadcast_3D_scalar
func @broadcast_3D_scalar(%arg0: tensor<i32>) -> tensor<3x2x4xi32> {
  %0 = "xla_hlo.broadcast"(%arg0) {broadcast_sizes = dense<[3, 2, 4]> : tensor<3xi64>} : (tensor<i32>) -> tensor<3x2x4xi32>
  return %0 : tensor<3x2x4xi32>
}
// CHECK: 3x2x4xi32={{\[}}[42 42 42 42][42 42 42 42{{\]}}]{{\[}}[42 42 42 42][42 42 42 42{{\]}}]{{\[}}[42 42 42 42][42 42 42 42{{\]}}]

// -----

// CHECK-LABEL: @broadcast_in_dim_3D_scalar
func @broadcast_in_dim_3D_scalar(%arg0: tensor<i32>) -> tensor<3x2x4xi32> {
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) : (tensor<i32>) -> tensor<3x2x4xi32>
  return %0 : tensor<3x2x4xi32>
}
// CHECK: 3x2x4xi32={{\[}}[42 42 42 42][42 42 42 42{{\]}}]{{\[}}[42 42 42 42][42 42 42 42{{\]}}]{{\[}}[42 42 42 42][42 42 42 42{{\]}}]
