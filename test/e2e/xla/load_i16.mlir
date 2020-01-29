// RUN: iree-run-mlir -iree-hal-target-backends=interpreter-bytecode %s -input-value="17xi16= 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17" | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s -input-value="17xi16= 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17" | IreeFileCheck %s)

// CHECK-LABEL: EXEC @load_i16
func @load_i16(%arg : tensor<17xi16>) -> tensor<17xi32> {
  %0 = "xla_hlo.convert"(%arg) : (tensor<17xi16>) -> tensor<17xi32>
  return %0 : tensor<17xi32>
}
// CHECK: 17xi32=1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
