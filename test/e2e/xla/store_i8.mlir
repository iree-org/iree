// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s --input_values="17xi32=-8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8" --output_types=i | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --target_backends=vulkan-spirv %s --input_values="17xi32=-8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8" --output_types=i | IreeFileCheck %s)

// CHECK-LABEL: EXEC @store_i8
func @store_i8(%arg : tensor<17xi32>) -> tensor<17xi8> {
  %0 = "xla_hlo.convert"(%arg) : (tensor<17xi32>) -> tensor<17xi8>
  return %0 : tensor<17xi8>
}
// CHECK: 17xi8=-8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8
