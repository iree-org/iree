// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @select
func @select() -> tensor<4xf32> {
  // TODO(b/132205704) support i1 in constants and function signatures.
  %input = iree.unfoldable_constant dense<[1, 0, 1, 0]> : tensor<4xi32>
  %zeros = iree.unfoldable_constant dense<0> : tensor<4xi32>
  %cond = "xla_hlo.compare"(%input, %zeros) {comparison_direction = "GT"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  %lhs = iree.unfoldable_constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %rhs = iree.unfoldable_constant dense<[5.0, 6.0, 7.0, 8.0]> : tensor<4xf32>
  %result = "xla_hlo.select"(%cond, %lhs, %rhs) : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %result : tensor<4xf32>
}
// CHECK: 4xf32=1 6 3 8
