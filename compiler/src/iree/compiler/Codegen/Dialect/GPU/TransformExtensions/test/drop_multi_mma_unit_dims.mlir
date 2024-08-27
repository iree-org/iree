// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @drop_multi_mma_unit_dims(%lhs: vector<1x1x4xf16>, %rhs: vector<1x1x4xf16>, %acc: vector<1x1x4xf32>) -> vector<1x1x4xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } : vector<1x1x4xf16>, vector<1x1x4xf16> into vector<1x1x4xf32>
  return %0 : vector<1x1x4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.drop_multi_mma_unit_dims
    } : !transform.any_op
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] =  affine_map<() -> ()>

// CHECK-LABEL: func @drop_multi_mma_unit_dims
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<1x1x4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<1x1x4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<1x1x4xf32>
//       CHECK:   %[[LHS_EXT:.+]] = vector.extract %[[LHS]][0, 0] : vector<4xf16> from vector<1x1x4xf16>
//       CHECK:   %[[RHS_EXT:.+]] = vector.extract %[[RHS]][0, 0] : vector<4xf16> from vector<1x1x4xf16>
//       CHECK:   %[[ACC_EXT:.+]] = vector.extract %[[ACC]][0, 0] : vector<4xf32> from vector<1x1x4xf32>
//       CHECK:   %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_EXT]], %[[RHS_EXT]], %[[ACC_EXT]]
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types = []
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>} : vector<4xf16>, vector<4xf16> into vector<4xf32>
//       CHECK:   vector.broadcast %[[MMA]] : vector<4xf32> to vector<1x1x4xf32>

// -----

#contraction_accesses = [
 affine_map<(i) -> (i)>,
 affine_map<(i) -> ()>,
 affine_map<(i) -> (i)>
]
func.func @drop_multi_mma_unit_dims_no_kn(%lhs: vector<1x4xf16>, %rhs: vector<4xf16>, %acc: vector<1x4xf32>) -> vector<1x4xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } : vector<1x4xf16>, vector<4xf16> into vector<1x4xf32>
  return %0 : vector<1x4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.drop_multi_mma_unit_dims
    } : !transform.any_op
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] =  affine_map<() -> ()>

// CHECK-LABEL: func @drop_multi_mma_unit_dims_no_kn
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<1x4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<1x4xf32>
//       CHECK:   %[[LHS_EXT:.+]] = vector.extract %[[LHS]][0] : vector<4xf16> from vector<1x4xf16>
//       CHECK:   %[[ACC_EXT:.+]] = vector.extract %[[ACC]][0] : vector<4xf32> from vector<1x4xf32>
//       CHECK:   %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_EXT]], %[[RHS]], %[[ACC_EXT]]
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types = []
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>} : vector<4xf16>, vector<4xf16> into vector<4xf32>
//       CHECK:   vector.broadcast %[[MMA]] : vector<4xf32> to vector<1x4xf32>
