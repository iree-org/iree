// RUN: iree-opt -split-input-file -iree-vmla-conversion -canonicalize %s | IreeFileCheck %s

// -----

func @f(%arg0: tensor<3x4xf32>, %arg1: tensor<4x5xf32>) -> tensor<3x5xf32> attributes {sym_visibility = "private"} {
  // CHECK: vmla.batch.matmul
  %0 = "xla_hlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = {
    lhs_batching_dimensions = dense<[]> : tensor<0xi64>,
    lhs_contracting_dimensions = dense<[1]> : tensor<1xi64>,
    rhs_batching_dimensions = dense<[]> : tensor<0xi64>,
    rhs_contracting_dimensions = dense<[0]> : tensor<1xi64>
  }} : (tensor<3x4xf32>, tensor<4x5xf32>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}
