// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-gpu-concretize-mma-shapes{concretize-result=false}, canonicalize, cse))' --split-input-file | FileCheck %s -check-prefixes=CHECK,CHECK-INPUTS
// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-gpu-concretize-mma-shapes{concretize-inputs=false}, canonicalize, cse))' --split-input-file | FileCheck %s -check-prefixes=CHECK,CHECK-RESULT

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @concretize_multi_mma_F32_16x16x16_F16(%lhs: tensor<2x2x16x16xf16>, %rhs: tensor<2x2x16x16xf16>, %acc: tensor<2x2x16x16xf32>) -> tensor<2x2x16x16xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } : tensor<2x2x16x16xf16>, tensor<2x2x16x16xf16> into tensor<2x2x16x16xf32>
  return %0 : tensor<2x2x16x16xf32>
}

// CHECK-LABEL:       func @concretize_multi_mma_F32_16x16x16_F16
// CHECK-SAME:          %[[LHS:[A-Za-z0-9]+]]: tensor<2x2x16x16xf16>
// CHECK-SAME:          %[[RHS:[A-Za-z0-9]+]]: tensor<2x2x16x16xf16>
// CHECK-SAME:          %[[ACC:[A-Za-z0-9]+]]: tensor<2x2x16x16xf32>

// CHECK-INPUTS-DAG:    %[[EXPANDED_LHS:.+]] = tensor.expand_shape %[[LHS]] {{\[}}[0], [1], [2], [3, 4]] output_shape [2, 2, 16, 4, 4] : tensor<2x2x16x16xf16> into tensor<2x2x16x4x4xf16>
// CHECK-INPUTS-DAG:    %[[EXPANDED_RHS:.+]] = tensor.expand_shape %[[RHS]] {{\[}}[0], [1], [2, 3], [4]] output_shape [2, 2, 4, 4, 16] : tensor<2x2x16x16xf16> into tensor<2x2x4x4x16xf16>
// CHECK-INPUTS:        %[[MMA:.+]] = iree_gpu.multi_mma %[[EXPANDED_LHS]], %[[EXPANDED_RHS]], %[[ACC]]
// CHECK-INPUTS-SAME:     : tensor<2x2x16x4x4xf16>, tensor<2x2x4x4x16xf16> into tensor<2x2x16x16xf32>
// CHECK-INPUTS:        return %[[MMA]]

// CHECK-RESULT-DAG:    %[[EXPANDED_ACC:.+]] = tensor.expand_shape %[[ACC]] {{\[}}[0], [1], [2, 3], [4]] output_shape [2, 2, 4, 4, 16] : tensor<2x2x16x16xf32> into tensor<2x2x4x4x16xf32>
// CHECK-RESULT:        %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS]], %[[RHS]], %[[EXPANDED_ACC]]
// CHECK-RESULT-SAME:     : tensor<2x2x16x16xf16>, tensor<2x2x16x16xf16> into tensor<2x2x4x4x16xf32>
// CHECK-RESULT:        %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMA]] {{\[}}[0], [1], [2, 3], [4]] : tensor<2x2x4x4x16xf32> into tensor<2x2x16x16xf32>
// CHECK-RESULT:        return %[[COLLAPSED]]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @concretize_multi_mma_I32_16x16x32_I8(%lhs: tensor<2x2x16x32xi8>, %rhs: tensor<2x2x16x32xi8>, %acc: tensor<2x2x16x16xi32>) -> tensor<2x2x16x16xi32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
    rhs_permutation = array<i64: 1, 0>
  } : tensor<2x2x16x32xi8>, tensor<2x2x16x32xi8> into tensor<2x2x16x16xi32>
  return %0 : tensor<2x2x16x16xi32>
}

// CHECK-LABEL:       func @concretize_multi_mma_I32_16x16x32_I8
// CHECK-SAME:          %[[LHS:[A-Za-z0-9]+]]: tensor<2x2x16x32xi8>
// CHECK-SAME:          %[[RHS:[A-Za-z0-9]+]]: tensor<2x2x16x32xi8>
// CHECK-SAME:          %[[ACC:[A-Za-z0-9]+]]: tensor<2x2x16x16xi32>

// CHECK-INPUTS-DAG:    %[[EXPANDED_LHS:.+]] = tensor.expand_shape %[[LHS]] {{\[}}[0], [1], [2], [3, 4]] output_shape [2, 2, 16, 4, 8] : tensor<2x2x16x32xi8> into tensor<2x2x16x4x8xi8>
// CHECK-INPUTS-DAG:    %[[EXPANDED_RHS:.+]] = tensor.expand_shape %[[RHS]] {{\[}}[0], [1], [2], [3, 4]] output_shape [2, 2, 16, 4, 8] : tensor<2x2x16x32xi8> into tensor<2x2x16x4x8xi8>
// CHECK-INPUTS:        %[[MMA:.+]] = iree_gpu.multi_mma %[[EXPANDED_LHS]], %[[EXPANDED_RHS]], %[[ACC]]
// CHECK-INPUTS-SAME:     : tensor<2x2x16x4x8xi8>, tensor<2x2x16x4x8xi8> into tensor<2x2x16x16xi32>
// CHECK-INPUTS:        return %[[MMA]]

// CHECK-RESULT-DAG:    %[[EXPANDED_ACC:.+]] = tensor.expand_shape %[[ACC]] {{\[}}[0], [1], [2, 3], [4]] output_shape [2, 2, 4, 4, 16] : tensor<2x2x16x16xi32> into tensor<2x2x4x4x16xi32>
// CHECK-RESULT:        %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS]], %[[RHS]], %[[EXPANDED_ACC]]
// CHECK-RESULT-SAME:     : tensor<2x2x16x32xi8>, tensor<2x2x16x32xi8> into tensor<2x2x4x4x16xi32>
// CHECK-RESULT:        %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMA]] {{\[}}[0], [1], [2, 3], [4]] : tensor<2x2x4x4x16xi32> into tensor<2x2x16x16xi32>
// CHECK-RESULT:        return %[[COLLAPSED]]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @concretize_multi_mma_F32_32x32x8_F16(%lhs: tensor<2x2x32x8xf16>, %rhs: tensor<2x2x8x32xf16>, %acc: tensor<2x2x32x32xf32>) -> tensor<2x2x32x32xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>
  } : tensor<2x2x32x8xf16>, tensor<2x2x8x32xf16> into tensor<2x2x32x32xf32>
  return %0 : tensor<2x2x32x32xf32>
}

// CHECK-LABEL:       func @concretize_multi_mma_F32_32x32x8_F16
// CHECK-SAME:          %[[LHS:[A-Za-z0-9]+]]: tensor<2x2x32x8xf16>
// CHECK-SAME:          %[[RHS:[A-Za-z0-9]+]]: tensor<2x2x8x32xf16>
// CHECK-SAME:          %[[ACC:[A-Za-z0-9]+]]: tensor<2x2x32x32xf32>

// CHECK-INPUTS-DAG:    %[[EXPANDED_LHS:.+]] = tensor.expand_shape %[[LHS]] {{\[}}[0], [1], [2], [3, 4]] output_shape [2, 2, 32, 2, 4] : tensor<2x2x32x8xf16> into tensor<2x2x32x2x4xf16>
// CHECK-INPUTS-DAG:    %[[EXPANDED_RHS:.+]] = tensor.expand_shape %[[RHS]] {{\[}}[0], [1], [2, 3], [4]] output_shape [2, 2, 2, 4, 32] : tensor<2x2x8x32xf16> into tensor<2x2x2x4x32xf16>
// CHECK-INPUTS:        %[[MMA:.+]] = iree_gpu.multi_mma %[[EXPANDED_LHS]], %[[EXPANDED_RHS]], %[[ACC]]
// CHECK-INPUTS-SAME:     : tensor<2x2x32x2x4xf16>, tensor<2x2x2x4x32xf16> into tensor<2x2x32x32xf32>
// CHECK-INPUTS:        return %[[MMA]]

// CHECK-RESULT-DAG:    %[[EXPANDED_ACC:.+]] = tensor.expand_shape %[[ACC]] {{\[}}[0], [1], [2, 3, 4], [5]] output_shape [2, 2, 4, 2, 4, 32] : tensor<2x2x32x32xf32> into tensor<2x2x4x2x4x32xf32>
// CHECK-RESULT:        %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS]], %[[RHS]], %[[EXPANDED_ACC]]
// CHECK-RESULT-SAME:     : tensor<2x2x32x8xf16>, tensor<2x2x8x32xf16> into tensor<2x2x4x2x4x32xf32>
// CHECK-RESULT:        %[[COLLAPSED:.+]] = tensor.collapse_shape %[[MMA]] {{\[}}[0], [1], [2, 3, 4], [5]] : tensor<2x2x4x2x4x32xf32> into tensor<2x2x32x32xf32>
// CHECK-RESULT:        return %[[COLLAPSED]]
