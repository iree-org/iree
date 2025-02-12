// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @distribute_multi_mma_F16_16x16x16_F32(%lhs: tensor<2x2x16x16xf16>, %rhs: tensor<2x2x16x16xf16>, %acc: tensor<2x2x16x16xf32>) -> tensor<2x2x16x16xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } : tensor<2x2x16x16xf16>, tensor<2x2x16x16xf16> into tensor<2x2x16x16xf32>
  return %0 : tensor<2x2x16x16xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %multi_mma = transform.structured.match ops{["iree_gpu.multi_mma"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.iree.distribute_multi_mma %multi_mma : (!transform.any_op) -> !transform.any_op
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
     transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    transform.yield
  }
}

// CHECK-LABEL: func @distribute_multi_mma_F16_16x16x16_F32
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<2x2x16x16xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<2x2x16x16xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: tensor<2x2x16x16xf32>
//       CHECK:   scf.forall (%[[LANE_ID:.+]]) in (64) shared_outs(%[[ITER_ARG:.+]] = %[[ACC]]) -> (tensor<2x2x16x16xf32>)
//       CHECK:     %[[ID:.+]]:3 = affine.delinearize_index %[[LANE_ID]] into (4, 16)
//       CHECK:     %[[ID1:.+]]  = affine.linearize_index disjoint [%[[ID]]#1, %c0] by (4, 4)
//       CHECK:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0, %[[ID]]#2, %[[ID1]]]
//  CHECK-SAME:       [2, 2, 1, 4] [1, 1, 1, 1] : tensor<2x2x16x16xf16> to tensor<2x2x1x4xf16>
//       CHECK:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0, %[[ID1]], %[[ID]]#2]
//  CHECK-SAME:       [2, 2, 4, 1] [1, 1, 1, 1] : tensor<2x2x16x16xf16> to tensor<2x2x4x1xf16>
//       CHECK:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ITER_ARG]][0, 0, %[[ID1]], %[[ID]]#2]
//  CHECK-SAME:       [2, 2, 4, 1] [1, 1, 1, 1] : tensor<2x2x16x16xf32> to tensor<2x2x4x1xf32>
//       CHECK:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]]
//  CHECK-SAME:       : tensor<2x2x1x4xf16>, tensor<2x2x4x1xf16> into tensor<2x2x4x1xf32>
//       CHECK:     scf.forall.in_parallel
//       CHECK:       tensor.parallel_insert_slice %[[MMA]] into %[[ITER_ARG]][0, 0, %[[ID1]], %[[ID]]#2]
//  CHECK-SAME:         [2, 2, 4, 1] [1, 1, 1, 1] : tensor<2x2x4x1xf32> into tensor<2x2x16x16xf32>
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (j, k)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @distribute_multi_mma_I8_16x16x32_I32(%lhs: tensor<2x2x16x32xi8>, %rhs: tensor<2x2x16x32xi8>, %acc: tensor<2x2x16x16xi32>) -> tensor<2x2x16x16xi32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_I32_16x16x32_I8>,
    rhs_permutation = array<i64: 1, 0>
  } : tensor<2x2x16x32xi8>, tensor<2x2x16x32xi8> into tensor<2x2x16x16xi32>
  return %0 : tensor<2x2x16x16xi32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %multi_mma = transform.structured.match ops{["iree_gpu.multi_mma"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.iree.distribute_multi_mma %multi_mma : (!transform.any_op) -> !transform.any_op
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
     transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    transform.yield
  }
}
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @distribute_multi_mma_I8_16x16x32_I32
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<2x2x16x32xi8>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<2x2x16x32xi8>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: tensor<2x2x16x16xi32>
//       CHECK:   scf.forall (%[[LANE_ID:.+]]) in (64) shared_outs(%[[ITER_ARG:.+]] = %[[ACC]]) -> (tensor<2x2x16x16xi32>)
//       CHECK:     %[[ID:.+]]:3  = affine.delinearize_index %[[LANE_ID]] into (4, 16)
//       CHECK:     %[[ID1:.+]] = affine.linearize_index disjoint [%[[ID]]#1, %c0] by (4, 8)
//       CHECK:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0, %[[ID]]#2, %[[ID1]]]
//  CHECK-SAME:       [2, 2, 1, 8] [1, 1, 1, 1] : tensor<2x2x16x32xi8> to tensor<2x2x1x8xi8>
//       CHECK:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0, %[[ID]]#2, %[[ID1]]]
//  CHECK-SAME:       [2, 2, 1, 8] [1, 1, 1, 1] : tensor<2x2x16x32xi8> to tensor<2x2x1x8xi8>
//       CHECK:     %[[ID1_2:.+]] = affine.linearize_index disjoint [%[[ID]]#1, %c0] by (4, 4)
//       CHECK:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ITER_ARG]][0, 0, %[[ID1_2]], %[[ID]]#2]
//  CHECK-SAME:       [2, 2, 4, 1] [1, 1, 1, 1] : tensor<2x2x16x16xi32> to tensor<2x2x4x1xi32>
//       CHECK:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]]
//  CHECK-SAME:       : tensor<2x2x1x8xi8>, tensor<2x2x1x8xi8> into tensor<2x2x4x1xi32>
//       CHECK:     scf.forall.in_parallel
//       CHECK:       tensor.parallel_insert_slice %[[MMA]] into %[[ITER_ARG]][0, 0, %[[ID1_2]], %[[ID]]#2]
//  CHECK-SAME:         [2, 2, 4, 1] [1, 1, 1, 1] : tensor<2x2x4x1xi32> into tensor<2x2x16x16xi32>
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]
