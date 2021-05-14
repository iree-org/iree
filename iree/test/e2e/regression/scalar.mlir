// RUN: iree-run-mlir -export-all -iree-hal-target-backends=vmvx %s | IreeFileCheck %s
// RUN: [[ $IREE_LLVMAOT_DISABLE == 1 ]] || (iree-run-mlir -export-all -iree-hal-target-backends=dylib-llvm-aot %s | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -export-all -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @scalar
func @scalar() -> i32 {
  %result = iree.unfoldable_constant 42 : i32
  return %result : i32
}
// CHECK: i32=42
