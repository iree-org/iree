// RUN: iree-run-mlir %s -iree-hal-target-backends=vmla | IreeFileCheck %s
// RUN: iree-run-mlir %s -iree-hal-target-backends=llvm-ir | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @main
func @main() -> tensor<5x5xf32> {
  %lhs = iree.unfoldable_constant dense<[
    [15.0, 14.0, 13.0],
    [12.0, 11.0, 10.0],
    [09.0, 08.0, 07.0],
    [06.0, 05.0, 04.0],
    [03.0, 02.0, 01.0]
  ]> : tensor<5x3xf32>
  %rhs = iree.unfoldable_constant dense<[
    [15.0, 14.0, 13.0, 12.0, 11.0],
    [10.0, 09.0, 08.0, 07.0, 06.0],
    [05.0, 04.0, 03.0, 02.0, 01.0]
  ]> : tensor<3x5xf32>
  %res = "xla_hlo.dot"(%lhs, %rhs) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>
  return %res : tensor<5x5xf32>
}

// CHECK: 5x5xf32=[430 388 346 304 262][340 307 274 241 208][250 226 202 178 154][160 145 130 115 100][70 64 58 52 46]
