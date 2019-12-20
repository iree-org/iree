// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --input_values=1xi32=42 --output_types=i | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --target_backends=vulkan-spirv %s --input_values=1xi32=42 --output_types=i | IreeFileCheck %s)

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
// CHECK: 1xi64=42

