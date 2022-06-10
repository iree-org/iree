// RUN: (iree-compile --iree-hal-target-backends=vmvx %s | iree-run-module --device=local-task --entry_function=abs --function_input=f32=-2) | FileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || ((iree-compile --iree-hal-target-backends=vulkan-spirv %s | iree-run-module --device=vulkan --entry_function=abs --function_input=f32=-2) | FileCheck %s)
// RUN: (iree-compile --iree-hal-target-backends=dylib-llvm-aot %s | iree-run-module --device=local-task --entry_function=abs --function_input=f32=-2) | FileCheck %s

// CHECK-LABEL: EXEC @abs
func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
  %result = math.abs %input : tensor<f32>
  return %result : tensor<f32>
}
// CHECK: f32=2
