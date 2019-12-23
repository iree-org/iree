// RUN: iree-run-mlir --target_backends=interpreter-bytecode --output_types=i %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --target_backends=vulkan-spirv --output_types=i %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @xla_constant_i8
func @xla_constant_i8() -> tensor<3xi8> {
  %0 = xla_hlo.constant dense<1> : tensor<3xi8>
  return %0 : tensor<3xi8>
}
// CHECK: 3xi8=1 1 1
