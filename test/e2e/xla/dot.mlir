// RUN: iree-run-mlir --target_backends=interpreter-bytecode -input_values="2xf32=0.3, 0.5" %s | IreeFileCheck %s

// CHECK-LABEL: EXEC @dot_passthrough
func @dot_passthrough(%arg0: tensor<2xf32>) -> tensor<1x3xf32> {
  %cst_1 = constant  dense<[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]> : tensor<2x3xf32>
  %1 = "xla_hlo.reshape"(%arg0) : (tensor<2xf32>) -> tensor<1x2xf32>
  %2 = "xla_hlo.dot"(%1, %cst_1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
  return %2 : tensor<1x3xf32>
}

// CHECK: 1x3xf32=[0.23 0.31 0.39]

// CHECK-LABEL: EXEC @dot_general_lower
func @dot_general_lower(%arg0: tensor<2xf32>) -> tensor<1x1x3xf32> {
  %cst_1 = constant  dense<[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]> : tensor<2x3xf32>
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<2xf32>) -> tensor<1x1x2xf32>
  %1 = "xla_hlo.dot_general"(%0, %cst_1) {dot_dimension_numbers = {lhs_batching_dimensions = dense<[]> : tensor<0xi64>, lhs_contracting_dimensions = dense<2> : tensor<1xi64>, rhs_batching_dimensions = dense<[]> : tensor<0xi64>, rhs_contracting_dimensions = dense<0> : tensor<1xi64>}, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<1x1x2xf32>, tensor<2x3xf32>) -> tensor<1x1x3xf32>
  return %1 : tensor<1x1x3xf32>
}

// CHECK:      1x1x3xf32=[
// CHECK-SAME: [0.23 0.31 0.39]
// CHECK-SAME: ]

// CHECK-LABEL: EXEC @dot_general_lower_swapped
func @dot_general_lower_swapped(%arg0: tensor<2xf32>) -> tensor<3x1x1xf32> {
  %cst_1 = constant  dense<[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]> : tensor<2x3xf32>
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<2xf32>) -> tensor<1x1x2xf32>
  %1 = "xla_hlo.dot_general"(%cst_1, %0) {dot_dimension_numbers = {lhs_batching_dimensions = dense<[]> : tensor<0xi64>, lhs_contracting_dimensions = dense<0> : tensor<1xi64>, rhs_batching_dimensions = dense<[]> : tensor<0xi64>, rhs_contracting_dimensions = dense<2> : tensor<1xi64>}, precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<2x3xf32>, tensor<1x1x2xf32>) -> tensor<3x1x1xf32>
  return %1 : tensor<3x1x1xf32>
}

// CHECK:      3x1x1xf32=[
// CHECK-SAME: [0.23]][
// CHECK-SAME: [0.31]][
// CHECK-SAME: [0.39]]
