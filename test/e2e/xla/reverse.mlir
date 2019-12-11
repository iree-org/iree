// RUN: iree-run-mlir --target_backends=interpreter-bytecode %s | IreeFileCheck %s
// RUN: iree-run-mlir --target_backends=vulkan-spirv %s | IreeFileCheck %s

// -----

// CHECK-LABEL: EXEC @xla_reverse
func @xla_reverse () -> (tensor<2x3xf32>, tensor <2x3xf32>, tensor <2x3xf32>) {
  %t1 = xla_hlo.constant dense<[[1.0e0, 2.0e0, 3.0e0], [4.0e0, 5.0e0, 6.0e0]]> : tensor<2x3xf32>
  %0 = "xla_hlo.reverse"(%t1) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %1 = "xla_hlo.reverse"(%t1) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  %2 = "xla_hlo.reverse"(%t1) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0, %1, %2: tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>
}
// CHECK: 2x3xf32=[4 5 6][1 2 3]
// CHECK-NEXT: 2x3xf32=[3 2 1][6 5 4]
// CHECK_NEXT: 2x3xf32=[6 5 4][3 2 1]
