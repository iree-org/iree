// RUN: iree-opt --iree-stablehlo-input-transformation-pipeline %s \
// RUN:   | FileCheck %s --implicit-check-not=stablehlo.

// The generic einsum lowering handles reductions that occur on only one
// operand, including when the other operand is scalar.
// CHECK-LABEL: func.func @scalar_lhs_independent_reduction
// CHECK: linalg.generic
func.func @scalar_lhs_independent_reduction(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  %cst = stablehlo.constant dense<1.0> : tensor<f32>
  %0 = "stablehlo.einsum"(%cst, %arg0) {einsum_config = ",ab->a"} : (tensor<f32>, tensor<4x8xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// Equal counts of unrelated reductions are still independent reductions.
// CHECK-LABEL: func.func @independent_reductions
// CHECK: linalg.generic
func.func @independent_reductions(%arg0: tensor<2x3xf32>, %arg1: tensor<4x5xf32>) -> tensor<2x4xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "ab,cd->ac"} : (tensor<2x3xf32>, tensor<4x5xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// Repeated labels require diagonal extraction before contraction.
// CHECK-LABEL: func.func @repeated_label_contraction
// CHECK: linalg.generic
func.func @repeated_label_contraction(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<f32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "ii,ii->"} : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// Dimensions are paired by label even when batch and contraction orders differ.
// CHECK-LABEL: func.func @reordered_contracting_and_batch
func.func @reordered_contracting_and_batch(%arg0: tensor<2x3x7x4x5xf32>, %arg1: tensor<5x3x2x6x4xf32>) -> tensor<2x3x7x6xf32> {
  %0 = "stablehlo.einsum"(%arg0, %arg1) {einsum_config = "bcijk,kcblj->bcil"} : (tensor<2x3x7x4x5xf32>, tensor<5x3x2x6x4xf32>) -> tensor<2x3x7x6xf32>
  return %0 : tensor<2x3x7x6xf32>
}

// Dynamic matvec and vector dot must remain legal through the full input
// pipeline, including the dot-vector preprocessing.
// CHECK-LABEL: func.func @dynamic_matvec
func.func @dynamic_matvec(%lhs: tensor<?x?xf32>, %rhs: tensor<?xf32>) -> tensor<?xf32> {
  %0 = "stablehlo.einsum"(%lhs, %rhs) {einsum_config = "ij,j->i"} : (tensor<?x?xf32>, tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func.func @dynamic_vector_dot
func.func @dynamic_vector_dot(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>) -> tensor<f32> {
  %0 = "stablehlo.einsum"(%lhs, %rhs) {einsum_config = "i,i->"} : (tensor<?xf32>, tensor<?xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}


// Full-pipeline bounds lowering precedes einsum specialisation.
// CHECK-LABEL: @bounded_transpose
// CHECK-NOT: #stablehlo.bounds
// CHECK: linalg.matmul
// CHECK: return
func.func @bounded_transpose(%lhs: tensor<?x3xf32, #stablehlo.bounds<7, ?>>, %rhs: tensor<3x?xf32, #stablehlo.bounds<?, 5>>) -> tensor<?x?xf32, #stablehlo.bounds<5, 7>> {
  %r = stablehlo.einsum %lhs, %rhs, config = "ij,jk->ki" : (tensor<?x3xf32, #stablehlo.bounds<7, ?>>, tensor<3x?xf32, #stablehlo.bounds<?, 5>>) -> tensor<?x?xf32, #stablehlo.bounds<5, 7>>
  return %r : tensor<?x?xf32, #stablehlo.bounds<5, 7>>
}
