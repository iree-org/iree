// RUN: iree-translate -iree-hal-target-backends=vmvx -iree-mlir-to-vm-bytecode-module %s | iree-benchmark-module --driver=vmvx --entry_function=abs --function_input=f32=-2 | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-translate --iree-hal-target-backends=vulkan-spirv -iree-mlir-to-vm-bytecode-module %s | iree-benchmark-module --driver=vulkan --entry_function=abs --function_input=f32=-2 | IreeFileCheck %s)
// RUN: [[ $IREE_LLVMAOT_DISABLE == 1 ]] || (iree-translate --iree-hal-target-backends=dylib-llvm-aot -iree-mlir-to-vm-bytecode-module %s | iree-benchmark-module --driver=dylib --entry_function=abs --function_input=f32=-2 | IreeFileCheck %s)

// CHECK-LABEL: BM_abs
func @abs(%input : tensor<f32>) -> (tensor<f32>) {
  %result = math.abs %input : tensor<f32>
  return %result : tensor<f32>
}
