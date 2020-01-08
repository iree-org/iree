// RUN: iree-run-mlir -iree-hal-target-backends=interpreter-bytecode %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @pad
func @pad_test() -> tensor<4x13xi32> {
  %input = iree.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %c0 = iree.unfoldable_constant dense<0> : tensor<i32>
  %res = "xla_hlo.pad"(%input, %c0) {
    edge_padding_low = dense<[0, 1]> : tensor<2xi64>,
    edge_padding_high = dense<[1, 5]> : tensor<2xi64>,
    interior_padding = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<2x3xi32>, tensor<i32>) -> tensor<4x13xi32>
  return %res : tensor<4x13xi32>
}

// CHECK-NEXT: 4x13xi32=
// CHECK-SAME: [0 1 0 0 2 0 0 3 0 0 0 0 0]
// CHECK-SAME: [0 0 0 0 0 0 0 0 0 0 0 0 0]
// CHECK-SAME: [0 4 0 0 5 0 0 6 0 0 0 0 0]
// CHECK-SAME: [0 0 0 0 0 0 0 0 0 0 0 0 0]

// -----

// CHECK-LABEL: EXEC @pad_no_op
func @pad_no_op() -> tensor<2x3xi32>
     attributes { iree.module.export } {
  %input = iree.unfoldable_constant dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
  %c0 = iree.unfoldable_constant dense<0> : tensor<i32>
  %res = "xla_hlo.pad"(%input, %c0) {edge_padding_high = dense<[0, 0]> : tensor<2xi64>, edge_padding_low = dense<[0, 0]> : tensor<2xi64>, interior_padding = dense<0> : tensor<2xi64>} : (tensor<2x3xi32>, tensor<i32>) -> tensor<2x3xi32>
  return %res : tensor<2x3xi32>
}

// CHECK-NEXT: 2x3xi32=
// CHECK-SAME: [1 2 3][4 5 6]
