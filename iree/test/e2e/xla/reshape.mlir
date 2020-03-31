// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s | IreeFileCheck %s
// RUN: [[ $IREE_LLVMJIT_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=llvm-ir %s | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @reshape_1D_2D
func @reshape_1D_2D() -> tensor<3x4xf32> {
  %0 = iree.unfoldable_constant dense<
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]>
       : tensor<12xf32>
  %result = "xla_hlo.reshape"(%0) : (tensor<12xf32>) -> tensor<3x4xf32>
  return %result : tensor<3x4xf32>
}
// CHECK: 3x4xf32=[1 2 3 4][5 6 7 8][9 10 11 12]

// -----

// CHECK-LABEL: EXEC @reshape_1D_3D
func @reshape_1D_3D() -> tensor<2x2x3xf32> {
  %0 = iree.unfoldable_constant dense<
       [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]>
       : tensor<12xf32>
  %result = "xla_hlo.reshape"(%0) : (tensor<12xf32>) -> tensor<2x2x3xf32>
  return %result : tensor<2x2x3xf32>
}
// CHECK 2x2x3xf32=\[[1 2 3][4 5 6]]\[[7 8 9][10 11 12]]
