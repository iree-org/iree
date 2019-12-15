// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --input_values="12xf32=[1 2 3 4 5 6 7 8 9 10 11 12]" | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --target_backends=vulkan-spirv %s --input_values="12xf32=[1 2 3 4 5 6 7 8 9 10 11 12]" | IreeFileCheck %s)

// CHECK-LABEL: EXEC @reshape_1D_2D
func @reshape_1D_2D(%arg : tensor<12xf32>) -> tensor<3x4xf32> {
  %result = "xla_hlo.reshape"(%arg) : (tensor<12xf32>) -> tensor<3x4xf32>
  return %result : tensor<3x4xf32>
}
// CHECK: 3x4xf32=[1 2 3 4][5 6 7 8][9 10 11 12]

// CHECK-LABEL: EXEC @reshape_1D_3D
func @reshape_1D_3D(%arg : tensor<12xf32>) -> tensor<2x2x3xf32> {
  %result = "xla_hlo.reshape"(%arg) : (tensor<12xf32>) -> tensor<2x2x3xf32>
  return %result : tensor<2x2x3xf32>
}
// CHECK 2x2x3xf32=\[[1 2 3][4 5 6]]\[[7 8 9][10 11 12]]
