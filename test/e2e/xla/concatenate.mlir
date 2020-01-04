// RUN: iree-run-mlir -iree-hal-target-backends=interpreter-bytecode %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @xla_concatenate
func @xla_concatenate() -> (tensor<2x5xi32>, tensor<2x5xi32>, tensor<2x7xi32>, tensor<4x2xi32>) {
  %c0 = iree.unfoldable_constant dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>
  %c1 = iree.unfoldable_constant dense<[[5, 6, 7], [8, 9, 10]]> : tensor<2x3xi32>
  %c2 = iree.unfoldable_constant dense<[[11, 12], [13, 14]]> : tensor<2x2xi32>
  %0 = "xla_hlo.concatenate"(%c0, %c1) {dimension = 1} : (tensor<2x2xi32>, tensor<2x3xi32>) -> tensor<2x5xi32>
  %1 = "xla_hlo.concatenate"(%c1, %c0) {dimension = 1} : (tensor<2x3xi32>, tensor<2x2xi32>) -> tensor<2x5xi32>
  %2 = "xla_hlo.concatenate"(%c0, %c1, %c2) {dimension = 1} : (tensor<2x2xi32>, tensor<2x3xi32>, tensor<2x2xi32>) -> tensor<2x7xi32>
  %3 = "xla_hlo.concatenate"(%c0, %c2) {dimension = 0} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<4x2xi32>
  return %0, %1, %2, %3: tensor<2x5xi32>, tensor<2x5xi32>, tensor<2x7xi32>, tensor<4x2xi32>
}
// CHECK:      2x5xi32=[1 2 5 6 7][3 4 8 9 10]
// CHECK-NEXT: 2x5xi32=[5 6 7 1 2][8 9 10 3 4]
// CHECK-NEXT: 2x7xi32=[1 2 5 6 7 11 12][3 4 8 9 10 13 14]
// CHECK-NEXT: 4x2xi32=[1 2][3 4][11 12][13 14]
