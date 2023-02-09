// RUN: iree-opt --split-input-file --verify-diagnostics --iree-mhlo-to-mhlo-preprocessing %s | FileCheck %s

func.func @dot_general_to_dot(%arg0: tensor<1x32x128x4xf32>, %arg1: tensor<128x4x8x64xf32>) -> tensor<1x32x8x64xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [],
      lhs_contracting_dimensions = [2, 3],
      rhs_batching_dimensions = [],
      rhs_contracting_dimensions = [0, 1],
    >,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<1x32x128x4xf32>, tensor<128x4x8x64xf32>) -> tensor<1x32x8x64xf32>
  return %0 : tensor<1x32x8x64xf32>
}

// CHECK: dot_general_to_dot(%[[ARG0:.+]]: tensor<1x32x128x4xf32>, %[[ARG1:.+]]: tensor<128x4x8x64xf32>) -> tensor<1x32x8x64xf32>
// CHECK: %[[ARG0_RESHAPED:.+]] = mhlo.reshape %[[ARG0]] : (tensor<1x32x128x4xf32>) -> tensor<32x512xf32>
// CHECK: %[[ARG1_RESHAPED:.+]] = mhlo.reshape %[[ARG1]] : (tensor<128x4x8x64xf32>) -> tensor<512x512xf32>
// CHECK: %[[DOT:.+]] = "mhlo.dot"(%[[ARG0_RESHAPED]], %[[ARG1_RESHAPED]])
// CHECK: %[[RESULT:.+]] = mhlo.reshape %[[DOT]] : (tensor<32x512xf32>) -> tensor<1x32x8x64xf32>
// CHECK: return %[[RESULT]] : tensor<1x32x8x64xf32>

// -----

func.func @dot_general_to_dot_general_rank_reduced(%arg0: tensor<1x8x32x64xf32>, %arg1 : tensor<1x8x64x32xf32>) -> tensor<1x8x32x32xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_batching_dimensions = [0, 1],
      rhs_contracting_dimensions = [2],
    >,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<1x8x32x64xf32>, tensor<1x8x64x32xf32>) -> tensor<1x8x32x32xf32>
  return %0 : tensor<1x8x32x32xf32>
}
// CHECK: dot_general_to_dot_general_rank_reduced(%[[ARG0:.+]]: tensor<1x8x32x64xf32>, %[[ARG1:.+]]: tensor<1x8x64x32xf32>) -> tensor<1x8x32x32xf32>
// CHECK: %[[ARG0_RESHAPED:.+]] = mhlo.reshape %[[ARG0]] : (tensor<1x8x32x64xf32>) -> tensor<8x32x64xf32>
// CHECK: %[[ARG1_RESHAPED:.+]] = mhlo.reshape %[[ARG1]] : (tensor<1x8x64x32xf32>) -> tensor<8x64x32xf32>
// CHECK: %[[DOT_RESULT:.+]] = "mhlo.dot_general"(%[[ARG0_RESHAPED]], %[[ARG1_RESHAPED]])
// CHECK: %[[RESULT:.+]] = mhlo.reshape %[[DOT_RESULT]] : (tensor<8x32x32xf32>) -> tensor<1x8x32x32xf32>
// CHECK: return %[[RESULT]] : tensor<1x8x32x32xf32>

// -----

func.func @dot_general_to_dot_general_rank_reduced_attribute(%arg0: tensor<1x8x32x64xf32>, %arg1 : tensor<1x8x64x32xf32>) -> tensor<1x8x32x32xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    unknown_attribute_to_propagate,
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_batching_dimensions = [0, 1],
      rhs_contracting_dimensions = [2],
    >,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<1x8x32x64xf32>, tensor<1x8x64x32xf32>) -> tensor<1x8x32x32xf32>
  return %0 : tensor<1x8x32x32xf32>
}
// CHECK: dot_general_to_dot_general_rank_reduced_attribute(%[[ARG0:.+]]: tensor<1x8x32x64xf32>, %[[ARG1:.+]]: tensor<1x8x64x32xf32>) -> tensor<1x8x32x32xf32>
// CHECK: %[[ARG0_RESHAPED:.+]] = mhlo.reshape %[[ARG0]] : (tensor<1x8x32x64xf32>) -> tensor<8x32x64xf32>
// CHECK: %[[ARG1_RESHAPED:.+]] = mhlo.reshape %[[ARG1]] : (tensor<1x8x64x32xf32>) -> tensor<8x64x32xf32>
// CHECK: %[[DOT_RESULT:.+]] = "mhlo.dot_general"(%[[ARG0_RESHAPED]], %[[ARG1_RESHAPED]]) {{{.*}}, unknown_attribute_to_propagate
// CHECK: %[[RESULT:.+]] = mhlo.reshape %[[DOT_RESULT]] : (tensor<8x32x32xf32>) -> tensor<1x8x32x32xf32>
// CHECK: return %[[RESULT]] : tensor<1x8x32x32xf32>

// -----

func.func @dot_general_to_dot_general_rank_reduced_a_transposed(%arg0: tensor<1x8x64x32xf32>, %arg1: tensor<1x8x64x32xf32>) -> tensor<1x8x32x32xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0, 1],
      rhs_contracting_dimensions = [2],
    >,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<1x8x64x32xf32>, tensor<1x8x64x32xf32>) -> tensor<1x8x32x32xf32>
  return %0 : tensor<1x8x32x32xf32>
}
// CHECK: dot_general_to_dot_general_rank_reduced_a_transposed(%[[ARG0:.+]]: tensor<1x8x64x32xf32>, %[[ARG1:.+]]: tensor<1x8x64x32xf32>) -> tensor<1x8x32x32xf32>
// CHECK: %[[ARG0_RESHAPED_TR:.+]] = "mhlo.transpose"(%[[ARG0]]) {permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>} : (tensor<1x8x64x32xf32>) -> tensor<1x8x32x64xf32>
// CHECK: %[[ARG0_RESHAPED:.+]] = mhlo.reshape %[[ARG0_RESHAPED_TR]] : (tensor<1x8x32x64xf32>) -> tensor<8x32x64xf32>
// CHECK: %[[ARG1_RSSHAPED:.+]] = mhlo.reshape %[[ARG1]] : (tensor<1x8x64x32xf32>) -> tensor<8x64x32xf32>
// CHECK: %[[DOT_RESULT:.+]] = "mhlo.dot_general"(%[[ARG0_RESHAPED]], %[[ARG1_RSSHAPED]])
// CHECK: %[[RESULT:.+]] = mhlo.reshape %[[DOT_RESULT]] : (tensor<8x32x32xf32>) -> tensor<1x8x32x32xf32>

// -----

func.func @dot_general_to_dot_general_rank_reduced_b_transposed(%arg0: tensor<1x8x32x64xf32>, %arg1: tensor<1x8x32x64xf32>) -> tensor<1x8x32x32xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [3],
      rhs_batching_dimensions = [0, 1],
      rhs_contracting_dimensions = [3],
    >,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<1x8x32x64xf32>, tensor<1x8x32x64xf32>) -> tensor<1x8x32x32xf32>
  return %0 : tensor<1x8x32x32xf32>
}
// CHECK: dot_general_to_dot_general_rank_reduced_b_transposed(%[[ARG0:.+]]: tensor<1x8x32x64xf32>, %[[ARG1:.+]]: tensor<1x8x32x64xf32>) -> tensor<1x8x32x32xf32>
// CHECK: %[[ARG1_RESHAPED_TR:.+]] = "mhlo.transpose"(%[[ARG1]]) {permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>} : (tensor<1x8x32x64xf32>) -> tensor<1x8x64x32xf32>
// CHECK: %[[ARG0_RESHAPED:.+]] = mhlo.reshape %[[ARG0]] : (tensor<1x8x32x64xf32>) -> tensor<8x32x64xf32>
// CHECK: %[[ARG1_RESHAPED:.+]] = mhlo.reshape %[[ARG1_RESHAPED_TR]] : (tensor<1x8x64x32xf32>) -> tensor<8x64x32xf32>
// CHECK: %[[DOT_RESULT:.+]] = "mhlo.dot_general"(%[[ARG0_RESHAPED]], %[[ARG1_RESHAPED]])
// CHECK: %[[RESULT:.+]] = mhlo.reshape %[[DOT_RESULT]] : (tensor<8x32x32xf32>) -> tensor<1x8x32x32xf32>
// CHECK: return %[[RESULT]] : tensor<1x8x32x32xf32>


// -----

func.func @dot_general_to_dot_general_rank_reduced_ab_transposed(%arg0: tensor<1x8x64x32xf32>, %arg1: tensor<1x8x32x64xf32>) -> tensor<1x8x32x32xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0, 1],
      rhs_contracting_dimensions = [3],
    >,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<1x8x64x32xf32>, tensor<1x8x32x64xf32>) -> tensor<1x8x32x32xf32>
  return %0 : tensor<1x8x32x32xf32>
}
// CHECK: dot_general_to_dot_general_rank_reduced_ab_transposed(%[[ARG0:.+]]: tensor<1x8x64x32xf32>, %[[ARG1:.+]]: tensor<1x8x32x64xf32>) -> tensor<1x8x32x32xf32>
// CHECK: %[[ARG0_RESHAPED_TR:.+]] = "mhlo.transpose"(%[[ARG0]]) {permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>} : (tensor<1x8x64x32xf32>) -> tensor<1x8x32x64xf32>
// CHECK: %[[ARG1_RESHAPED_TR:.+]] = "mhlo.transpose"(%[[ARG1]]) {permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>} : (tensor<1x8x32x64xf32>) -> tensor<1x8x64x32xf32>
// CHECK: %[[ARG0_RESHAPED:.+]] = mhlo.reshape %[[ARG0_RESHAPED_TR]] : (tensor<1x8x32x64xf32>) -> tensor<8x32x64xf32>
// CHECK: %[[ARG1_RESHAPED:.+]] = mhlo.reshape %[[ARG1_RESHAPED_TR]] : (tensor<1x8x64x32xf32>) -> tensor<8x64x32xf32>
// CHECK: %[[DOT_RESULT:.+]] = "mhlo.dot_general"(%[[ARG0_RESHAPED]], %[[ARG1_RESHAPED]])
// CHECK: %[[RESULT:.+]] = mhlo.reshape %[[DOT_RESULT]] : (tensor<8x32x32xf32>) -> tensor<1x8x32x32xf32>
// CHECK: return %[[RESULT]] : tensor<1x8x32x32xf32>

// -----

func.func @dot_general_4d_transposed(%arg0: tensor<1x1x8x64xf32>, %arg1: tensor<1x512x8x64xf32>) -> tensor<1x8x1x512xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 2],
      lhs_contracting_dimensions = [3],
      rhs_batching_dimensions = [0, 2],
      rhs_contracting_dimensions = [3],
    >,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<1x1x8x64xf32>, tensor<1x512x8x64xf32>) -> tensor<1x8x1x512xf32>
  return %0 : tensor<1x8x1x512xf32>
}

// CHECK-LABEL: func.func @dot_general_4d_transposed
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK:         %[[ARG0_TRANSPOSED:.+]] = "mhlo.transpose"(%[[ARG0]])
// CHECK-SAME:      permutation = dense<[0, 2, 1, 3]>
// CHECK:         %[[ARG1_TRANSPOSED:.+]] = "mhlo.transpose"(%[[ARG1]])
// CHECK-SAME:      permutation = dense<[0, 2, 3, 1]>
// CHECK:         %[[ARG0_RESHAPED:.+]] = mhlo.reshape %[[ARG0_TRANSPOSED]] : (tensor<1x8x1x64xf32>) -> tensor<8x1x64xf32>
// CHECK:         %[[ARG1_RESHAPED:.+]] = mhlo.reshape %[[ARG1_TRANSPOSED]] : (tensor<1x8x64x512xf32>) -> tensor<8x64x512xf32>
// CHECK:         %[[DOT_RESULT:.+]] = "mhlo.dot_general"(%[[ARG0_RESHAPED]], %[[ARG1_RESHAPED]])
// CHECK:         %[[RESULT:.+]] = mhlo.reshape %[[DOT_RESULT]] : (tensor<8x1x512xf32>) -> tensor<1x8x1x512xf32>
// CHECK:         return %[[RESULT]] : tensor<1x8x1x512xf32>

// -----

func.func @dot_general_1d_batching_1d_contracting(%arg0: tensor<64x155x4x36xf32>, %arg1: tensor<309x4x36xf32>) -> tensor<4x64x155x309xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [2],
      rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2]
    >} : (tensor<64x155x4x36xf32>, tensor<309x4x36xf32>) -> tensor<4x64x155x309xf32>
  return %0 : tensor<4x64x155x309xf32>
}

// CHECK-LABEL: func.func @dot_general_1d_batching_1d_contracting
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]+]]
// CHECK: %[[ARG0_RESHAPED_TR:.+]] = "mhlo.transpose"(%[[ARG0]])
// CHECK-SAME: {permutation = dense<[2, 0, 1, 3]> : tensor<4xi64>}
// CHECK-SAME: (tensor<64x155x4x36xf32>) -> tensor<4x64x155x36xf32>
// CHECK: %[[ARG1_RESHAPED_TR:.+]] = "mhlo.transpose"(%[[ARG1]])
// CHECK-SAME: {permutation = dense<[1, 2, 0]> : tensor<3xi64>}
// CHECK-SAME: (tensor<309x4x36xf32>) -> tensor<4x36x309xf32>
// CHECK: %[[ARG0_RESHAPED:.+]] = mhlo.reshape %[[ARG0_RESHAPED_TR]]
// CHECK-SAME: (tensor<4x64x155x36xf32>) -> tensor<4x9920x36xf32>
// CHECK: %[[DOT_RESULT:.+]] = "mhlo.dot_general"(%[[ARG0_RESHAPED]], %[[ARG1_RESHAPED_TR]])
// CHECK-SAME: (tensor<4x9920x36xf32>, tensor<4x36x309xf32>) -> tensor<4x9920x309xf32>
// CHECK: mhlo.reshape %[[DOT_RESULT]] : (tensor<4x9920x309xf32>) -> tensor<4x64x155x309xf32>

// -----

// CHECK-LABEL: @dot_general_2d
func.func public @dot_general_2d(%arg0: tensor<4x3xf32> {mhlo.sharding = ""}, %arg1: tensor<4x3xf32> {mhlo.sharding = ""}) -> tensor<3xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [1], rhs_batching_dimensions = [1], lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision HIGHEST>, #mhlo<precision HIGHEST>]} : (tensor<4x3xf32>, tensor<4x3xf32>) -> tensor<3xf32>

  // CHECK: %[[LHS:.+]] = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<4x3xf32>) -> tensor<3x4xf32>
  // CHECK: %[[RHS:.+]] = "mhlo.transpose"(%arg1) {permutation = dense<[1, 0]> : tensor<2xi64>} : (tensor<4x3xf32>) -> tensor<3x4xf32>
  // CHECK: "mhlo.dot_general"(%[[LHS]], %[[RHS]])
  // CHECK-SAME: dot_dimension_numbers = #mhlo.dot<
  // CHECK-SAME: lhs_batching_dimensions = [0]
  // CHECK-SAME: rhs_batching_dimensions = [0]
  // CHECK-SAME: lhs_contracting_dimensions = [1]
  // CHECK-SAME: rhs_contracting_dimensions = [1]>
  // CHECK-SAME: precision_config = [#mhlo<precision HIGHEST>, #mhlo<precision HIGHEST>]
  return %0 : tensor<3xf32>
}

// -----

// CHECK-LABEL: @dot_general_fuse
func.func @dot_general_fuse(%arg0: tensor<64x155x4x36xf16>, %arg1: tensor<309x4x36xf16>) -> tensor<4x64x155x309xf32> {
  %0 = "mhlo.convert"(%arg0) : (tensor<64x155x4x36xf16>) -> tensor<64x155x4x36xf32>
  %1 = "mhlo.convert"(%arg1) : (tensor<309x4x36xf16>) -> tensor<309x4x36xf32>
  %2 = "mhlo.dot_general"(%0, %1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [2],
      rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [2]
    >} : (tensor<64x155x4x36xf32>, tensor<309x4x36xf32>) -> tensor<4x64x155x309xf32>
  // CHECK: "mhlo.dot_general"
  // CHECK-SAME: (tensor<4x9920x36xf16>, tensor<4x36x309xf16>) -> tensor<4x9920x309xf32>
  return %2 : tensor<4x64x155x309xf32>
}

// -----

// CHECK-LABEL: @dot_is_mul
func.func @dot_is_mul(%arg0: tensor<?x1xf16>, %arg1: tensor<1x?xf16>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[D0:.+]] = "mhlo.get_dimension_size"(%arg0) {dimension = 0 : i64}
  // CHECK-DAG: %[[D1:.+]] = "mhlo.get_dimension_size"(%arg1) {dimension = 1 : i64}
  // CHECK-DAG: %[[D0R:.+]] = mhlo.reshape %[[D0]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-DAG: %[[D1R:.+]] = mhlo.reshape %[[D1]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-DAG: %[[SZ:.+]] = "mhlo.concatenate"(%[[D0R]], %[[D1R]]) {dimension = 0 : i64}
  // CHECK-DAG: %[[L:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg0, %[[SZ]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}
  // CHECK-DAG: %[[R:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg1, %[[SZ]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}
  // CHECK-DAG: %[[CL:.+]] = mhlo.convert %[[L]] : (tensor<?x?xf16>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[CR:.+]] = mhlo.convert %[[R]] : (tensor<?x?xf16>) -> tensor<?x?xf32>
  // CHECK: %[[RESULT:.+]] = mhlo.multiply %[[CL]], %[[CR]] : tensor<?x?xf32>
  %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<?x1xf16>, tensor<1x?xf16>) -> tensor<?x?xf32>
  // CHECK: return %[[RESULT]]
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func.func @dot_general_to_mul
func.func @dot_general_to_mul(%arg0: tensor<?x3x1x5xf32>, %arg1: tensor<3x6x1x?xf32>) -> tensor<3x?x5x6x?xf32> {
  // CHECK-DAG: %[[THREE:.+]] = mhlo.constant dense<3> : tensor<1xi32>
  // CHECK-DAG: %[[FIVE:.+]] = mhlo.constant dense<5> : tensor<1xi32>
  // CHECK-DAG: %[[SIX:.+]] = mhlo.constant dense<6> : tensor<1xi32>
  // CHECK-DAG: %[[LT:.+]] = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>}
  // CHECK-DAG: %[[RT:.+]] = "mhlo.transpose"(%arg1) {permutation = dense<[0, 1, 3, 2]> : tensor<4xi64>}
  // CHECK-DAG: %[[LDIM:.+]] = "mhlo.get_dimension_size"(%[[LT]]) {dimension = 1 : i64}
  // CHECK-DAG: %[[LDIMR:.+]] = mhlo.reshape %[[LDIM]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-DAG: %[[LSHAPE:.+]] = "mhlo.concatenate"(%[[THREE]], %[[LDIMR]], %[[FIVE]]) {dimension = 0 : i64}
  // CHECK-DAG: %[[LRESHAPE:.+]] = mhlo.dynamic_reshape %[[LT]], %[[LSHAPE]] : (tensor<3x?x5x1xf32>, tensor<3xi32>) -> tensor<3x?x5xf32>
  // CHECK-DAG: %[[RDIM:.+]] = "mhlo.get_dimension_size"(%[[RT]]) {dimension = 2 : i64} : (tensor<3x6x?x1xf32>) -> tensor<i32>
  // CHECK-DAG: %[[RDIMR:.+]] = mhlo.reshape %[[RDIM]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-DAG: %[[RSHAPE:.+]] = "mhlo.concatenate"(%[[THREE]], %[[SIX]], %[[RDIMR]]) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  // CHECK-DAG: %[[RRESHAPE:.+]] = mhlo.dynamic_reshape %[[RT]], %[[RSHAPE]] : (tensor<3x6x?x1xf32>, tensor<3xi32>) -> tensor<3x6x?xf32>
  // CHECK-DAG: %[[OSHAPE:.+]] = "mhlo.concatenate"(%[[THREE]], %[[LDIMR]], %[[FIVE]], %[[SIX]], %[[RDIMR]]) {dimension = 0 : i64}
  // CHECK-DAG: %[[L:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[LRESHAPE]], %[[OSHAPE]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<3x?x5xf32>, tensor<5xi32>) -> tensor<3x?x5x6x?xf32>
  // CHECK-DAG: %[[R:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[RRESHAPE]], %[[OSHAPE]]) {broadcast_dimensions = dense<[0, 3, 4]> : tensor<3xi64>} : (tensor<3x6x?xf32>, tensor<5xi32>) -> tensor<3x?x5x6x?xf32>
  // CHECK: %[[MUL:.+]] = mhlo.multiply %[[L]], %[[R]]
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [1], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<?x3x1x5xf32>, tensor<3x6x1x?xf32>) -> tensor<3x?x5x6x?xf32>
  // CHECK: return %[[MUL]]
  func.return %0 : tensor<3x?x5x6x?xf32>
}

// -----

// CHECK-LABEL: func.func @dot_general_to_mul_cast
func.func @dot_general_to_mul_cast(%arg0: tensor<4x17xf16>, %arg1: tensor<4x309xf16>) -> tensor<4x17x309xf32> {
  // CHECK: %[[OSHAPE:.+]] = mhlo.constant dense<[4, 17, 309]> : tensor<3xi32>
  // CHECK: %[[LSHAPE:.+]] = mhlo.constant dense<[4, 17]> : tensor<2xi32>
  // CHECK: %[[RSHAPE:.+]] = mhlo.constant dense<[4, 309]> : tensor<2xi32>
  // CHECK: %[[LRESHAPE:.+]] = mhlo.dynamic_reshape %arg0, %[[LSHAPE]]
  // CHECK: %[[RRESHAPE:.+]] = mhlo.dynamic_reshape %arg1, %[[RSHAPE]]
  // CHECK: %[[LBROAD:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[LRESHAPE]], %[[OSHAPE]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}
  // CHECK: %[[RBROAD:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[RRESHAPE]], %[[OSHAPE]]) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>}
  // CHECK: %[[L:.+]] = mhlo.convert %[[LBROAD]] : (tensor<4x17x309xf16>) -> tensor<4x17x309xf32>
  // CHECK: %[[R:.+]] = mhlo.convert %[[RBROAD]] : (tensor<4x17x309xf16>) -> tensor<4x17x309xf32>
  // CHECK: %[[MUL:.+]] = mhlo.multiply %[[L]], %[[R]]
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0]>} : (tensor<4x17xf16>, tensor<4x309xf16>) -> tensor<4x17x309xf32>
  // CHECK: return %[[MUL]]
  return %0 : tensor<4x17x309xf32>
}
