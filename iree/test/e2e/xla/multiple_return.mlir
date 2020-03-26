// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s | IreeFileCheck %s
// RUN: [[ $IREE_LLVMJIT_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=llvm-ir %s | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv -iree-use-linalg-to-spirv-path %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @scalar
func @scalar() -> (i32, i32) {
  %result = iree.unfoldable_constant 42 : i32
  return %result, %result : i32, i32
}
// CHECK-COUNT-2: i32=42

// -----
// CHECK-LABEL: EXEC @rank0tensor
func @rank0tensor() -> (tensor<f32>, tensor<f32>) {
  %res = iree.unfoldable_constant dense<42.0> : tensor<f32>
  return %res, %res : tensor<f32>, tensor<f32>
}
// CHECK-COUNT-2: f32=42

// -----
// CHECK-LABEL: EXEC @tensor
func @tensor() -> (tensor<2x2xf32>, tensor<2x2xf32>) {
  %res = iree.unfoldable_constant
      dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  return %res, %res : tensor<2x2xf32>, tensor<2x2xf32>
}
// CHECK-COUNT-2: 2x2xf32=[1 2][3 4]

// -----
// CHECK-LABEL: EXEC @many_tensor
func @many_tensor() -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>,
                        tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) {
  %res = iree.unfoldable_constant
      dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  return %res, %res, %res, %res, %res, %res :
        tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>,
        tensor<2x2xf32>, tensor<2x2xf32>
}
// CHECK-COUNT-6: 2x2xf32=[1 2][3 4]
