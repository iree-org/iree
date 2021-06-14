// RUN: (iree-run-mlir --iree-hal-target-backends=vmvx --function-input="f32=-2" %s) | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv --function-input="f32=-2" %s | IreeFileCheck %s)
// RUN: [[ $IREE_LLVMAOT_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=dylib-llvm-aot --function-input="f32=-2" %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @abs
func @abs(%input : tensor<f32>) -> (tensor<f32>) {
  %result = absf %input : tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=2
