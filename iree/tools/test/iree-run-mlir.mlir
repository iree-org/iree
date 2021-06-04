// RUN: (iree-run-mlir --iree-input-type=mhlo --iree-hal-target-backends=vmvx --function-input="i32=-2" %s) | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --iree-input-type=mhlo -iree-hal-target-backends=vulkan-spirv --function-input="i32=-2" %s | IreeFileCheck %s)
// RUN: [[ $IREE_LLVMAOT_DISABLE == 1 ]] || (iree-run-mlir --iree-input-type=mhlo -iree-hal-target-backends=dylib-llvm-aot --function-input="i32=-2" %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @abs
func @abs(%input : tensor<i32>) -> (tensor<i32>) attributes { iree.module.export } {
  %result = "mhlo.abs"(%input) : (tensor<i32>) -> tensor<i32>
  return %result : tensor<i32>
}
// CHECK: i32=2
