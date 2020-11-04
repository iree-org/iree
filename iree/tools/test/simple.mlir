// iree-run-module
// RUN: (iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s | iree-run-module --driver=vmla --entry_function=abs --function_inputs="i32=-2") | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || ((iree-translate --iree-hal-target-backends=vulkan-spirv -iree-mlir-to-vm-bytecode-module %s | iree-run-module --driver=vulkan --entry_function=abs --function_inputs="i32=-2") | IreeFileCheck %s)
// RUN: [[ $IREE_LLVMJIT_DISABLE == 1 ]] || ((iree-translate --iree-hal-target-backends=llvm-ir -iree-mlir-to-vm-bytecode-module %s | iree-run-module --driver=llvm --entry_function=abs --function_inputs="i32=-2") | IreeFileCheck %s)

// iree-benchmark-module
// RUN: iree-translate --iree-hal-target-backends=vmla -iree-mlir-to-vm-bytecode-module %s | iree-benchmark-module --driver=vmla --entry_function=abs --function_inputs="i32=-2" | IreeFileCheck %s --check-prefix=BENCHMARK
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-translate --iree-hal-target-backends=vulkan-spirv -iree-mlir-to-vm-bytecode-module %s | iree-benchmark-module --driver=vulkan --entry_function=abs --function_inputs="i32=-2" | IreeFileCheck %s --check-prefix=BENCHMARK)
// RUN: [[ $IREE_LLVMJIT_DISABLE == 1 ]] || (iree-translate --iree-hal-target-backends=llvm-ir -iree-mlir-to-vm-bytecode-module %s | iree-benchmark-module --driver=llvm --entry_function=abs --function_inputs="i32=-2" | IreeFileCheck %s --check-prefix=BENCHMARK)

// iree-run-mlir
// RUN: (iree-run-mlir --iree-hal-target-backends=vmla --function-input="i32=-2" %s) | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv --function-input="i32=-2" %s | IreeFileCheck %s)
// RUN: [[ $IREE_LLVMJIT_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=llvm-ir --function-input="i32=-2" %s | IreeFileCheck %s)

// BENCHMARK-LABEL: BM_abs
// CHECK-LABEL: EXEC @abs
func @abs(%input : tensor<i32>) -> (tensor<i32>) attributes { iree.module.export } {
  %result = "mhlo.abs"(%input) : (tensor<i32>) -> tensor<i32>
  return %result : tensor<i32>
}
// CHECK: i32=2
