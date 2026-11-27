// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-stablehlo-preprocessing-einsum-to-dot-general))" %s \
// RUN:   | FileCheck %s

// Diagonals remain einsums for the generic Linalg lowering.
//
// CHECK-LABEL: func @einsum_diag
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
func.func @einsum_diag(%arg0: tensor<6x6xf32>) -> tensor<6xf32> {
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = "stablehlo.einsum"(%0, %arg0) {einsum_config = ",ii->i"} : (tensor<f32>, tensor<6x6xf32>) -> tensor<6xf32>
  func.return %1 : tensor<6xf32>
}
// CHECK:         %[[CST:.+]] = stablehlo.constant dense<{{.*}} : tensor<f32>
// CHECK:         stablehlo.einsum

// CHECK-LABEL: func @einsum_batched_matrix_high_rank_vector_mul
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
func.func @einsum_batched_matrix_high_rank_vector_mul(%arg0: tensor<8x2x6xf32>, %arg1: tensor<8x5x3x6xf32>) -> tensor<8x5x3x2xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "bxy,bijy->bijx"} : (tensor<8x2x6xf32>, tensor<8x5x3x6xf32>) -> tensor<8x5x3x2xf32>
  func.return %0 : tensor<8x5x3x2xf32>
}
// CHECK:         %[[DG:.+]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      batching_dims = [0] x [0]
// CHECK-SAME:      contracting_dims = [2] x [3]
// CHECK-SAME:    : (tensor<8x2x6xf32>, tensor<8x5x3x6xf32>) -> tensor<8x2x5x3xf32>
// CHECK:         %[[T:.+]] = stablehlo.transpose %[[DG]]
// CHECK-SAME:      dims = [0, 2, 3, 1]
// CHECK-SAME:    : (tensor<8x2x5x3xf32>) -> tensor<8x5x3x2xf32>

// CHECK-LABEL: func @matmul
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
func.func @matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "ij,jk->ik"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}
// CHECK:         %{{.+}} = stablehlo.dot_general %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      contracting_dims = [1] x [0]
// CHECK-SAME:    : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>

// CHECK-LABEL: func @matvec
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
func.func @matvec(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "ij,j->i"} : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK:         %{{.+}} = stablehlo.dot_general %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      contracting_dims = [1] x [0]
// CHECK-SAME:    : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?xf32>

// CHECK-LABEL: func @dot
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]+]]
func.func @dot(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<f32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "i,i->"} : (tensor<?xf32>, tensor<?xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK:         %{{.+}} = stablehlo.dot_general %[[ARG0]], %[[ARG1]]
// CHECK-SAME:      contracting_dims = [0] x [0]
// CHECK-SAME:    : (tensor<?xf32>, tensor<?xf32>) -> tensor<f32>


// The rewrite must decline independent reductions on either operand, including
// unrelated reductions with equal dimension counts.
// CHECK-LABEL: @independent_lhs_reduction
// CHECK: stablehlo.einsum
// CHECK-NOT: stablehlo.dot_general
func.func @independent_lhs_reduction(%lhs: tensor<2x3xf32>, %rhs: tensor<f32>) -> tensor<2xf32> {
  %r = stablehlo.einsum %lhs, %rhs, config = "ab,->a" : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: @independent_rhs_reduction
// CHECK: stablehlo.einsum
// CHECK-NOT: stablehlo.dot_general
func.func @independent_rhs_reduction(%lhs: tensor<f32>, %rhs: tensor<2x3xf32>) -> tensor<2xf32> {
  %r = stablehlo.einsum %lhs, %rhs, config = ",ab->a" : (tensor<f32>, tensor<2x3xf32>) -> tensor<2xf32>
  return %r : tensor<2xf32>
}

// CHECK-LABEL: @independent_reductions
// CHECK: stablehlo.einsum
// CHECK-NOT: stablehlo.dot_general
func.func @independent_reductions(%lhs: tensor<2x3xf32>, %rhs: tensor<4x5xf32>) -> tensor<2x4xf32> {
  %r = stablehlo.einsum %lhs, %rhs, config = "ab,cd->ac" : (tensor<2x3xf32>, tensor<4x5xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// CHECK-LABEL: @repeated_labels
// CHECK: stablehlo.einsum
// CHECK-NOT: stablehlo.dot_general
func.func @repeated_labels(%lhs: tensor<3x3xf32>, %rhs: tensor<3x3xf32>) -> tensor<f32> {
  %r = stablehlo.einsum %lhs, %rhs, config = "ii,ii->" : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<f32>
  return %r : tensor<f32>
}

// Both batch and contraction lists follow lhs label order on both operands.
// CHECK-LABEL: @reordered_shared_labels
// CHECK: stablehlo.dot_general
// CHECK-SAME: batching_dims = [0, 1] x [2, 1]
// CHECK-SAME: contracting_dims = [3, 4] x [4, 0]
func.func @reordered_shared_labels(%lhs: tensor<2x3x7x4x5xf32>, %rhs: tensor<5x3x2x6x4xf32>) -> tensor<2x3x7x6xf32> {
  %r = stablehlo.einsum %lhs, %rhs, config = "bcijk,kcblj->bcil" : (tensor<2x3x7x4x5xf32>, tensor<5x3x2x6x4xf32>) -> tensor<2x3x7x6xf32>
  return %r : tensor<2x3x7x6xf32>
}

// The result may be declared less precisely than its static operands.
// CHECK-LABEL: @dynamic_result_transpose
// CHECK: %[[DOT:.+]] = stablehlo.dot_general
// CHECK-SAME: -> tensor<?x?xf32>
// CHECK: stablehlo.transpose %[[DOT]], dims = [1, 0]
// CHECK-SAME: -> tensor<?x?xf32>
func.func @dynamic_result_transpose(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<?x?xf32> {
  %r = stablehlo.einsum %lhs, %rhs, config = "ij,jk->ki" : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// The generic lowering owns ellipsis expansion; preprocessing must not emit an
// error for a configuration it cannot represent with explicit labels.
// CHECK-LABEL: @ellipsis
// CHECK: stablehlo.einsum
// CHECK-NOT: stablehlo.dot_general
func.func @ellipsis(%lhs: tensor<2x3x4xf32>, %rhs: tensor<2x4x5xf32>) -> tensor<2x3x5xf32> {
  %r = stablehlo.einsum %lhs, %rhs, config = "...ij,...jk->...ik" : (tensor<2x3x4xf32>, tensor<2x4x5xf32>) -> tensor<2x3x5xf32>
  return %r : tensor<2x3x5xf32>
}

// Einsum has no semantic verifier. A standalone optional pass must decline a
// label/rank mismatch before inspecting dimension sizes.
// CHECK-LABEL: @negative_label_count
// CHECK: stablehlo.einsum
// CHECK-NOT: stablehlo.dot_general
func.func @negative_label_count(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  %r = stablehlo.einsum %lhs, %rhs, config = "i,jk->ik" : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// CHECK-LABEL: @negative_result_label
// CHECK: stablehlo.einsum
// CHECK-NOT: stablehlo.dot_general
func.func @negative_result_label(%lhs: tensor<2x3xf32>, %rhs: tensor<3x4xf32>) -> tensor<2x4xf32> {
  %r = stablehlo.einsum %lhs, %rhs, config = "ij,jk->iz" : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %r : tensor<2x4xf32>
}

// A standalone rewrite leaves encoded tensors intact. The full pipeline lowers
// bounds before preprocessing, so it can still specialise the contraction.
// CHECK-LABEL: @encoded_transpose
// CHECK: stablehlo.einsum
// CHECK-SAME: #stablehlo.bounds<5, 7>
// CHECK-NOT: stablehlo.dot_general
func.func @encoded_transpose(%lhs: tensor<?x3xf32, #stablehlo.bounds<7, ?>>, %rhs: tensor<3x?xf32, #stablehlo.bounds<?, 5>>) -> tensor<?x?xf32, #stablehlo.bounds<5, 7>> {
  %r = stablehlo.einsum %lhs, %rhs, config = "ij,jk->ki" : (tensor<?x3xf32, #stablehlo.bounds<7, ?>>, tensor<3x?xf32, #stablehlo.bounds<?, 5>>) -> tensor<?x?xf32, #stablehlo.bounds<5, 7>>
  return %r : tensor<?x?xf32, #stablehlo.bounds<5, 7>>
}
