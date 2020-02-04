// RUN: (iree-translate --iree-hal-target-backends=interpreter-bytecode -iree-mlir-to-vm-bytecode-module %s | iree-run-module --driver=interpreter --entry_function=abs --inputs="i32=-2") | IreeFileCheck %s

// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || ((iree-translate --iree-hal-target-backends=vulkan-spirv -iree-mlir-to-vm-bytecode-module %s | iree-run-module --driver=vulkan --entry_function=abs --inputs="i32=-2") | IreeFileCheck %s)

// RUN: (iree-run-mlir --iree-hal-target-backends=interpreter-bytecode --input-value="i32=-2" %s) | IreeFileCheck %s

// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv --input-value="i32=-2" %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @abs
func @abs(%input : tensor<i32>) -> (tensor<i32>) attributes { iree.module.export } {
  %result = "xla_hlo.abs"(%input) : (tensor<i32>) -> tensor<i32>
  return %result : tensor<i32>
}
// CHECK: i32=2
