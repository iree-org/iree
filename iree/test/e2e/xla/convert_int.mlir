// RUN: [[ $IREE_LLVMJIT_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vmla %s | IreeFileCheck %s)
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @narrow_int
func @narrow_int() -> tensor<1xi8> {
  %input = iree.unfoldable_constant dense<42> : tensor<1xi32>
  %0 = "xla_hlo.convert"(%input) : (tensor<1xi32>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}
// CHECK: 1xi8=42

// CHECK-LABEL: EXEC @widen_int
func @widen_int() -> tensor<1xi64> {
  %input = iree.unfoldable_constant dense<42> : tensor<1xi32>
  %0 = "xla_hlo.convert"(%input) : (tensor<1xi32>) -> tensor<1xi64>
  return %0 : tensor<1xi64>
}
// CHECK: 1xi32=42

