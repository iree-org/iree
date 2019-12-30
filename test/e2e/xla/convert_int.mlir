// RUN: iree-run-mlir -iree-hal-target-backends=interpreter-bytecode %s -input-value=1xi32=42 | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s -input-value=1xi32=42 | IreeFileCheck %s)

// CHECK-LABEL: EXEC @narrow_int
func @narrow_int(%arg : tensor<1xi32>) -> tensor<1xi8> {
  %0 = "xla_hlo.convert"(%arg) : (tensor<1xi32>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}
// CHECK: 1xi8=42

// CHECK-LABEL: EXEC @widen_int
func @widen_int(%arg : tensor<1xi32>) -> tensor<1xi64> {
  %0 = "xla_hlo.convert"(%arg) : (tensor<1xi32>) -> tensor<1xi64>
  return %0 : tensor<1xi64>
}
// CHECK: 1xi32=42

