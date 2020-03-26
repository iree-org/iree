// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s | IreeFileCheck %s
// RUN: [[ $IREE_LLVMJIT_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=llvm-ir %s | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @dot_passthrough
func @dot_passthrough() -> tensor<1x3xf32> {
  %lhs = iree.unfoldable_constant dense<[[0.3, 0.5]]> : tensor<1x2xf32>
  %rhs = iree.unfoldable_constant  dense<[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]> : tensor<2x3xf32>
  %res = "xla_hlo.dot"(%lhs, %rhs) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
  return %res : tensor<1x3xf32>
}

// CHECK: 1x3xf32=[0.23 0.31 0.39]
