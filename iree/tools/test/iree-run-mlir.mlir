// RUN: (iree-run-mlir --iree-hal-target-backends=vmvx --function-input="f32=-2" %s) | FileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv --function-input="f32=-2" %s | FileCheck %s)
// RUN: iree-run-mlir -iree-hal-target-backends=dylib-llvm-aot --function-input="f32=-2" %s | FileCheck %s

// CHECK-LABEL: EXEC @abs
func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
  %result = math.abs %input : tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=2
