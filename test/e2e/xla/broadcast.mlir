// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv -input-value="2x4xf32= 1 2 3 4 5 6 7 8" %s | IreeFileCheck %s)

// CHECK-LABEL: @broadcast_2D_3D
func @broadcast_2D_3D(%arg0: tensor<2x4xf32>) -> tensor<3x2x4xf32> {
  %0 = "xla_hlo.broadcast"(%arg0) {broadcast_sizes = dense<[3]> : tensor<1xi64>} : (tensor<2x4xf32>) -> tensor<3x2x4xf32>
  return %0 : tensor<3x2x4xf32>
}
// CHECK: 3x2x4xf32={{\[}}[1 2 3 4][5 6 7 8{{\]}}]{{\[}}[1 2 3 4][5 6 7 8{{\]}}]{{\[}}[1 2 3 4][5 6 7 8{{\]}}]

// -----

// CHECK-LABEL: @broadcast_in_dim_2D_3D
func @broadcast_in_dim_2D_3D(%arg0: tensor<2x4xf32>) -> tensor<3x2x4xf32> {
  %0 = "xla_hlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xf32>) -> tensor<3x2x4xf32>
  return %0 : tensor<3x2x4xf32>
}
// CHECK: 3x2x4xf32={{\[}}[1 2 3 4][5 6 7 8]{{\]}}{{\[}}[1 2 3 4][5 6 7 8]{{\]}}{{\[}}[1 2 3 4][5 6 7 8]{{\]}}
