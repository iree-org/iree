// RUN: iree-run-mlir -iree-hal-target-backends=interpreter-bytecode %s | IreeFileCheck %s
// RUN: [[ $IREE_VULKAN_DISABLE == 1 ]] || (iree-run-mlir -iree-hal-target-backends=vulkan-spirv %s | IreeFileCheck %s)

// CHECK-LABEL: EXEC @dot_passthrough
func @dot_passthrough() -> tensor<1x3xf32> {
  %lhs = iree.unfoldable_constant dense<[[0.3, 0.5]]> : tensor<1x2xf32>
  %rhs = iree.unfoldable_constant  dense<[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]> : tensor<2x3xf32>
  %res = "xla_hlo.dot"(%lhs, %rhs) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
  return %res : tensor<1x3xf32>
}

// CHECK: 1x3xf32=[0.23 0.31 0.39]

// -----

// CHECK-LABEL: EXEC @dot_general_lower
func @dot_general_lower() -> tensor<1x1x3xf32> {
  %lhs = iree.unfoldable_constant dense<[[[0.3, 0.5]]]> : tensor<1x1x2xf32>
  %rhs = iree.unfoldable_constant  dense<[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]> : tensor<2x3xf32>
  %res = "xla_hlo.dot_general"(%lhs, %rhs) {dot_dimension_numbers = {lhs_batching_dimensions = dense<[]> : tensor<0xi64>, lhs_contracting_dimensions = dense<2> : tensor<1xi64>, rhs_batching_dimensions = dense<[]> : tensor<0xi64>, rhs_contracting_dimensions = dense<0> : tensor<1xi64>}, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x1x2xf32>, tensor<2x3xf32>) -> tensor<1x1x3xf32>
  return %res : tensor<1x1x3xf32>
}

// CHECK:      1x1x3xf32=[
// CHECK-SAME:   [0.23 0.31 0.39]
// CHECK-SAME: ]

// -----

// CHECK-LABEL: EXEC @dot_general_lower_swapped
func @dot_general_lower_swapped() -> tensor<3x1x1xf32> {
  %lhs = iree.unfoldable_constant  dense<[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]> : tensor<2x3xf32>
  %rhs = iree.unfoldable_constant dense<[[[0.3, 0.5]]]> : tensor<1x1x2xf32>
  %res = "xla_hlo.dot_general"(%lhs, %rhs) {dot_dimension_numbers = {lhs_batching_dimensions = dense<[]> : tensor<0xi64>, lhs_contracting_dimensions = dense<0> : tensor<1xi64>, rhs_batching_dimensions = dense<[]> : tensor<0xi64>, rhs_contracting_dimensions = dense<2> : tensor<1xi64>}, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<2x3xf32>, tensor<1x1x2xf32>) -> tensor<3x1x1xf32>
  return %res : tensor<3x1x1xf32>
}

// CHECK:      3x1x1xf32=[
// CHECK-SAME: [0.23]][
// CHECK-SAME: [0.31]][
// CHECK-SAME: [0.39]]
