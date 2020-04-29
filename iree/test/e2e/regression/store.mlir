// RUN: iree-run-mlir -iree-hal-target-backends=vmla %s -input-value="17xi32=-8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8" | IreeFileCheck %s
// TODO(GH-1688): [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s -input-value="17xi32=-8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8" | IreeFileCheck %s)

// CHECK-LABEL: EXEC @store_i8
func @store_i8(%arg : tensor<17xi32>) -> tensor<17xi8> {
  %0 = "xla_hlo.convert"(%arg) : (tensor<17xi32>) -> tensor<17xi8>
  return %0 : tensor<17xi8>
}
// CHECK: 17xi8=-8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8

// -----

// CHECK-LABEL: EXEC @store_i16
func @store_i16(%arg : tensor<17xi32>) -> tensor<17xi16> {
  %0 = "xla_hlo.convert"(%arg) : (tensor<17xi32>) -> tensor<17xi16>
  return %0 : tensor<17xi16>
}
// CHECK: 17xi16=-8 -7 -6 -5 -4 -3 -2 -1 0 1 2 3 4 5 6 7 8
