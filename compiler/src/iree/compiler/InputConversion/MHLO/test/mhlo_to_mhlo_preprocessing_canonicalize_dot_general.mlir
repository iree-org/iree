// RUN: iree-opt --split-input-file --verify-diagnostics --iree-mhlo-to-mhlo-preprocessing %s | FileCheck %s

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

// CHECK-LABEL: @dot_general_4d
func.func public @dot_general_4d(%arg0: tensor<1x2x3xf32> {mhlo.sharding = ""}, %arg1: tensor<1x4x2x3xf32> {mhlo.sharding = ""}) -> tensor<1x2x4xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {dot_dimension_numbers = #mhlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 2], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [3]>, precision_config = [#mhlo<precision HIGHEST>, #mhlo<precision HIGHEST>]} : (tensor<1x2x3xf32>, tensor<1x4x2x3xf32>) -> tensor<1x2x4xf32>

  // CHECK: %[[RHS_T:.+]] = "mhlo.transpose"(%arg1) {permutation = dense<[0, 2, 3, 1]> : tensor<4xi64>} : (tensor<1x4x2x3xf32>) -> tensor<1x2x3x4xf32>
  // CHECK: %[[LHS_R:.+]] = mhlo.reshape %arg0 : (tensor<1x2x3xf32>) -> tensor<2x1x3xf32>
  // CHECK: %[[RHS_R:.+]] = mhlo.reshape %[[RHS_T]] : (tensor<1x2x3x4xf32>) -> tensor<2x3x4xf32> 
  // CHECK: %[[DOT:.+]] = "mhlo.dot_general"(%[[LHS_R]], %[[RHS_R]])
  // CHECK-SAME: dot_dimension_numbers = #mhlo.dot<
  // CHECK-SAME: lhs_batching_dimensions = [0]
  // CHECK-SAME: rhs_batching_dimensions = [0]
  // CHECK-SAME: lhs_contracting_dimensions = [2]
  // CHECK-SAME: rhs_contracting_dimensions = [1]>
  // CHECK-SAME: precision_config = [#mhlo<precision HIGHEST>, #mhlo<precision HIGHEST>]
  // CHECK: mhlo.reshape %[[DOT]] : (tensor<2x1x4xf32>) -> tensor<1x2x4xf32>
  return %0 : tensor<1x2x4xf32>
}
