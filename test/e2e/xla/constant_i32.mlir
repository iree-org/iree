// RUN: iree-run-mlir --target_backends=interpreter-bytecode --output_types=i,i %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir --target_backends=vulkan-spirv --output_types=i,i %s | IreeFileCheck %s)

// -----

// CHECK-LABEL: EXEC @xla_constant_i32
func @xla_constant_i32 () -> (tensor<2x2x3xi32>, tensor<2x2x3xi32>) {
  %0 = xla_hlo.constant dense<[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]> : tensor<2x2x3xi32>
  %1 = xla_hlo.constant dense<1> : tensor<2x2x3xi32>
  return %0, %1: tensor<2x2x3xi32>, tensor<2x2x3xi32>
}
// CHECK: 2x2x3xi32={{\[}}[1 2 3][4 5 6]]{{\[}}[7 8 9][10 11 12]]
// CHECK-NEXT: 2x2x3xi32={{\[}}[1 1 1][1 1 1]]{{\[}}[1 1 1][1 1 1]]
