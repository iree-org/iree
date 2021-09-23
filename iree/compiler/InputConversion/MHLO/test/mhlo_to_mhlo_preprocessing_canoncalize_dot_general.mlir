// RUN: iree-opt -split-input-file -verify-diagnostics -iree-mhlo-to-mhlo-preprocessing %s | IreeFileCheck %s

func @dot_general_to_dot(%arg0: tensor<1x32x128x4xf32>, %arg1: tensor<128x4x8x64xf32>) -> tensor<1x32x8x64xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [],
      lhs_contracting_dimensions = [2, 3],
      rhs_batching_dimensions = [],
      rhs_contracting_dimensions = [0, 1],
    >,
    precision_config = ["DEFAULT", "DEFAULT"]
  } : (tensor<1x32x128x4xf32>, tensor<128x4x8x64xf32>) -> tensor<1x32x8x64xf32>
  return %0 : tensor<1x32x8x64xf32>
}

// CHECK: dot_general_to_dot(%[[ARG0:.+]]: tensor<1x32x128x4xf32>, %[[ARG1:.+]]: tensor<128x4x8x64xf32>) -> tensor<1x32x8x64xf32>
// CHECK: %[[ARG0_RESHAPED:.+]] = "mhlo.reshape"(%[[ARG0]]) : (tensor<1x32x128x4xf32>) -> tensor<32x512xf32>
// CHECK: %[[ARG1_RESHAPED:.+]] = "mhlo.reshape"(%[[ARG1]]) : (tensor<128x4x8x64xf32>) -> tensor<512x512xf32>
// CHECK: %[[DOT:.+]] = "mhlo.dot"(%[[ARG0_RESHAPED]], %[[ARG1_RESHAPED]])
// CHECK: %[[RESULT:.+]] = "mhlo.reshape"(%[[DOT]]) : (tensor<32x512xf32>) -> tensor<1x32x8x64xf32>
// CHECK: return %[[RESULT]] : tensor<1x32x8x64xf32>

// -----

func @dot_general_to_dot_general_rank_reduced(%arg0: tensor<1x8x32x64xf32>, %arg1 : tensor<1x8x64x32xf32>) -> tensor<1x8x32x32xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_batching_dimensions = [0, 1],
      rhs_contracting_dimensions = [2],
    >,
    precision_config = ["DEFAULT", "DEFAULT"]
  } : (tensor<1x8x32x64xf32>, tensor<1x8x64x32xf32>) -> tensor<1x8x32x32xf32>
  return %0 : tensor<1x8x32x32xf32>
}
// CHECK: dot_general_to_dot_general_rank_reduced(%[[ARG0:.+]]: tensor<1x8x32x64xf32>, %[[ARG1:.+]]: tensor<1x8x64x32xf32>) -> tensor<1x8x32x32xf32>
// CHECK: %[[ARG0_RESHAPED:.+]] = "mhlo.reshape"(%[[ARG0]]) : (tensor<1x8x32x64xf32>) -> tensor<8x32x64xf32>
// CHECK: %[[ARG1_RESHAPED:.+]] = "mhlo.reshape"(%[[ARG1]]) : (tensor<1x8x64x32xf32>) -> tensor<8x64x32xf32>
// CHECK: %[[DOT_RESULT:.+]] = "mhlo.dot_general"(%[[ARG0_RESHAPED]], %[[ARG1_RESHAPED]])
// CHECK: %[[RESULT:.+]] = "mhlo.reshape"(%[[DOT_RESULT]]) : (tensor<8x32x32xf32>) -> tensor<1x8x32x32xf32>
// CHECK: return %[[RESULT]] : tensor<1x8x32x32xf32>

// -----

func @dot_general_to_dot_general_rank_reduced_a_transposed(%arg0: tensor<1x8x64x32xf32>, %arg1: tensor<1x8x64x32xf32>) -> tensor<1x8x32x32xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0, 1],
      rhs_contracting_dimensions = [2],
    >,
    precision_config = ["DEFAULT", "DEFAULT"]
  } : (tensor<1x8x64x32xf32>, tensor<1x8x64x32xf32>) -> tensor<1x8x32x32xf32>
  return %0 : tensor<1x8x32x32xf32>
}
// CHECK: dot_general_to_dot_general_rank_reduced_a_transposed(%[[ARG0:.+]]: tensor<1x8x64x32xf32>, %[[ARG1:.+]]: tensor<1x8x64x32xf32>) -> tensor<1x8x32x32xf32>
// CHECK: %[[ARG0_RESHAPED:.+]] = "mhlo.reshape"(%[[ARG0]]) : (tensor<1x8x64x32xf32>) -> tensor<8x64x32xf32>
// CHECK: %[[ARG1_RSSHAPED:.+]] = "mhlo.reshape"(%[[ARG1]]) : (tensor<1x8x64x32xf32>) -> tensor<8x64x32xf32>
// CHECK: %[[ARG0_RESHAPED_TR:.+]] = "mhlo.transpose"(%[[ARG0_RESHAPED]]) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<8x64x32xf32>) -> tensor<8x32x64xf32>
// CHECK: %[[DOT_RESULT:.+]] = "mhlo.dot_general"(%[[ARG0_RESHAPED_TR]], %[[ARG1_RSSHAPED]])
// CHECK: %[[RESULT:.+]] = "mhlo.reshape"(%[[DOT_RESULT]]) : (tensor<8x32x32xf32>) -> tensor<1x8x32x32xf32>

// -----

func @dot_general_to_dot_general_rank_reduced_b_transposed(%arg0: tensor<1x8x32x64xf32>, %arg1: tensor<1x8x32x64xf32>) -> tensor<1x8x32x32xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_batching_dimensions = [0,1 ],
      rhs_contracting_dimensions = [3],
    >,
    precision_config = ["DEFAULT", "DEFAULT"]
  } : (tensor<1x8x32x64xf32>, tensor<1x8x32x64xf32>) -> tensor<1x8x32x32xf32>
  return %0 : tensor<1x8x32x32xf32>
}
// CHECK: dot_general_to_dot_general_rank_reduced_b_transposed(%[[ARG0:.+]]: tensor<1x8x32x64xf32>, %[[ARG1:.+]]: tensor<1x8x32x64xf32>) -> tensor<1x8x32x32xf32>
// CHECK: %[[ARG0_RESHAPED:.+]] = "mhlo.reshape"(%[[ARG0]]) : (tensor<1x8x32x64xf32>) -> tensor<8x32x64xf32>
// CHECK: %[[ARG1_RESHAPED:.+]] = "mhlo.reshape"(%[[ARG1]]) : (tensor<1x8x32x64xf32>) -> tensor<8x32x64xf32>
// CHECK: %[[ARG1_RESHAPED_TR:.+]] = "mhlo.transpose"(%[[ARG1_RESHAPED]]) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<8x32x64xf32>) -> tensor<8x64x32xf32>
// CHECK: %[[DOT_RESULT:.+]] = "mhlo.dot_general"(%[[ARG0_RESHAPED]], %[[ARG1_RESHAPED_TR]])
// CHECK: %[[RESULT:.+]] = "mhlo.reshape"(%[[DOT_RESULT]]) : (tensor<8x32x32xf32>) -> tensor<1x8x32x32xf32>
// CHECK: return %[[RESULT]] : tensor<1x8x32x32xf32>


// -----

func @dot_general_to_dot_general_rank_reduced_ab_transposed(%arg0: tensor<1x8x64x32xf32>, %arg1: tensor<1x8x32x64xf32>) -> tensor<1x8x32x32xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0, 1],
      rhs_contracting_dimensions = [3],
    >,
    precision_config = ["DEFAULT", "DEFAULT"]
  } : (tensor<1x8x64x32xf32>, tensor<1x8x32x64xf32>) -> tensor<1x8x32x32xf32>
  return %0 : tensor<1x8x32x32xf32>
}
// CHECK: dot_general_to_dot_general_rank_reduced_ab_transposed(%[[ARG0:.+]]: tensor<1x8x64x32xf32>, %[[ARG1:.+]]: tensor<1x8x32x64xf32>) -> tensor<1x8x32x32xf32>
// CHECK: %[[ARG0_RESHAPED:.+]] = "mhlo.reshape"(%[[ARG0]]) : (tensor<1x8x64x32xf32>) -> tensor<8x64x32xf32>
// CHECK: %[[ARG1_RESHAPED:.+]] = "mhlo.reshape"(%[[ARG1]]) : (tensor<1x8x32x64xf32>) -> tensor<8x32x64xf32>
// CHECK: %[[ARG0_RESHAPED_TR:.+]] = "mhlo.transpose"(%[[ARG0_RESHAPED]]) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<8x64x32xf32>) -> tensor<8x32x64xf32>
// CHECK: %[[ARG1_RESHAPED_TR:.+]] = "mhlo.transpose"(%[[ARG1_RESHAPED]]) {permutation = dense<[0, 2, 1]> : tensor<3xi64>} : (tensor<8x32x64xf32>) -> tensor<8x64x32xf32>
// CHECK: %[[DOT_RESULT:.+]] = "mhlo.dot_general"(%[[ARG0_RESHAPED_TR]], %[[ARG1_RESHAPED_TR]])
// CHECK: %[[RESULT:.+]] = "mhlo.reshape"(%[[DOT_RESULT]]) : (tensor<8x32x32xf32>) -> tensor<1x8x32x32xf32>
// CHECK: return %[[RESULT]] : tensor<1x8x32x32xf32>

// -----

func @dot_general_4d_transposed(%arg0: tensor<1x1x8x64xf32>, %arg1: tensor<1x512x8x64xf32>) -> tensor<1x8x1x512xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 2],
      lhs_contracting_dimensions = [3],
      rhs_batching_dimensions = [0, 2],
      rhs_contracting_dimensions = [3],
    >,
    precision_config = ["DEFAULT", "DEFAULT"]
  } : (tensor<1x1x8x64xf32>, tensor<1x512x8x64xf32>) -> tensor<1x8x1x512xf32>
  return %0 : tensor<1x8x1x512xf32>
}

// CHECK-LABEL: func @dot_general_4d_transposed
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %[[ARG0_TRANSPOSED:.+]] = "mhlo.transpose"(%[[ARG0]])
// CHECK-SAME:      permutation = dense<[0, 2, 1, 3]>
// CHECK:         %[[ARG1_TRANSPOSED:.+]] = "mhlo.transpose"(%[[ARG1]])
// CHECK-SAME:      permutation = dense<[0, 2, 3, 1]>
// CHECK:         %[[ARG0_RESHAPED:.+]] = "mhlo.reshape"(%[[ARG0_TRANSPOSED]]) : (tensor<1x8x1x64xf32>) -> tensor<8x1x64xf32>
// CHECK:         %[[ARG1_RESHAPED:.+]] = "mhlo.reshape"(%[[ARG1_TRANSPOSED]]) : (tensor<1x8x64x512xf32>) -> tensor<8x64x512xf32>
// CHECK:         %[[DOT_RESULT:.+]] = "mhlo.dot_general"(%[[ARG0_RESHAPED]], %[[ARG1_RESHAPED]])
// CHECK:         %[[RESULT:.+]] = "mhlo.reshape"(%[[DOT_RESULT]]) : (tensor<8x1x512xf32>) -> tensor<1x8x1x512xf32>
// CHECK:         return %[[RESULT]] : tensor<1x8x1x512xf32>
