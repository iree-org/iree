// RUN: (iree-translate --iree-hal-target-backends=interpreter-bytecode -iree-mlir-to-vm-bytecode-module %s | iree-run-module --entry_function=multi_input) | IreeFileCheck %s

// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || ((iree-translate --iree-hal-target-backends=vulkan-spirv -iree-mlir-to-vm-bytecode-module %s | iree-run-module --driver=vulkan --entry_function=multi_input) | IreeFileCheck %s)

// RUN: (iree-run-mlir --iree-hal-target-backends=interpreter-bytecode %s) | IreeFileCheck %s

// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv  %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @multi_input
func @multi_input() -> i32 attributes { iree.module.export } {
  %c = iree.unfoldable_constant 42 : i32
  return %c : i32
}
// CHECK: i32=42
