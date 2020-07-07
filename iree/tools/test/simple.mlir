// iree-run-module
// RUN: (iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s | iree-run-module --driver=vmla --entry_function=abs --inputs="i32=-2") | IreeFileCheck %s

// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || ((iree-translate --iree-hal-target-backends=vulkan-spirv -iree-mlir-to-vm-bytecode-module %s | iree-run-module --driver=vulkan --entry_function=abs --inputs="i32=-2") | IreeFileCheck %s)

// iree-benchmark-module (only checking exit codes).
// RUN: iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s -o ${TEST_TMPDIR?}/bc.module && iree-benchmark-module --driver=vmla --entry_function=abs --inputs="i32=-2" --input_file=${TEST_TMPDIR?}/bc.module

// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s -o ${TEST_TMPDIR?}/bc.module && iree-benchmark-module --driver=vmla --entry_function=abs --inputs="i32=-2" --input_file=${TEST_TMPDIR?}/bc.module)

// iree-run-mlir
// RUN: (iree-run-mlir --iree-hal-target-backends=vmla --input-value="i32=-2" %s) | IreeFileCheck %s

// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv --input-value="i32=-2" %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @abs
func @abs(%input : tensor<i32>) -> (tensor<i32>) attributes { iree.module.export } {
  %result = "mhlo.abs"(%input) : (tensor<i32>) -> tensor<i32>
  return %result : tensor<i32>
}
// CHECK: i32=2
