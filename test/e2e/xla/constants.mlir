// RUN: iree-run-mlir -iree-hal-target-backends=interpreter-bytecode %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @xla_constant_i32
func @xla_constant_i32 () -> (tensor<2x2x3xi32>, tensor<2x2x3xi32>) {
  %0 = iree.unfoldable_constant dense<[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]> : tensor<2x2x3xi32>
  %1 = iree.unfoldable_constant dense<1> : tensor<2x2x3xi32>
  return %0, %1: tensor<2x2x3xi32>, tensor<2x2x3xi32>
}
// CHECK: 2x2x3xi32={{\[}}[1 2 3][4 5 6]]{{\[}}[7 8 9][10 11 12]]
// CHECK-NEXT: 2x2x3xi32={{\[}}[1 1 1][1 1 1]]{{\[}}[1 1 1][1 1 1]]

// -----

// CHECK-LABEL: EXEC @xla_constant_f32
func @xla_constant_f32 () -> (tensor<2x2x3xf32>, tensor<2x2x3xf32>) {
  %0 = iree.unfoldable_constant dense<[[[1.1e0, 2.1e0, 3.1e0], [4.1e0, 5.1e0, 6.1e0]], [[7.1e0, 8.1e0, 9.1e0], [10.1e0, 11.1e0, 12.1e0]]]> : tensor<2x2x3xf32>
  %1 = iree.unfoldable_constant dense<1.1e0> : tensor<2x2x3xf32>
  return %0, %1: tensor<2x2x3xf32>, tensor<2x2x3xf32>
}
// CHECK: 2x2x3xf32={{\[}}[1.1 2.1 3.1][4.1 5.1 6.1]]{{\[}}[7.1 8.1 9.1][10.1 11.1 12.1]]
// CHECK-NEXT: 2x2x3xf32={{\[}}[1.1 1.1 1.1][1.1 1.1 1.1]]{{\[}}[1.1 1.1 1.1][1.1 1.1 1.1]]

// -----

// CHECK-LABEL: EXEC @xla_constant_i8
func @xla_constant_i8() -> tensor<3xi8> {
  %0 = iree.unfoldable_constant dense<1> : tensor<3xi8>
  return %0 : tensor<3xi8>
}
// CHECK: 3xi8=1 1 1
