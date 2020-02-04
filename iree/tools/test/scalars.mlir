// RUN: (iree-translate --iree-hal-target-backends=interpreter-bytecode -iree-mlir-to-vm-bytecode-module %s | iree-run-module --entry_function=scalar --inputs="i32=42") | IreeFileCheck %s

// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || ((iree-translate --iree-hal-target-backends=vulkan-spirv -iree-mlir-to-vm-bytecode-module %s | iree-run-module --driver=vulkan --entry_function=scalar --inputs="i32=42") | IreeFileCheck %s)

// RUN: (iree-run-mlir --iree-hal-target-backends=interpreter-bytecode --input-value=i32=42 %s) | IreeFileCheck %s

// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv --input-value=i32=42 %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @scalar
func @scalar(%arg0 : i32) -> i32 attributes { iree.module.export } {
  return %arg0 : i32
}
// CHECK: i32=42
