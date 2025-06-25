// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @drop_inner_tiled_unit_dims(%lhs: vector<1x1x4xf16>, %rhs: vector<1x1x4xf16>, %acc: vector<1x1x4xf32>) -> vector<1x1x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } : vector<1x1x4xf16>, vector<1x1x4xf16> into vector<1x1x4xf32>
  return %0 : vector<1x1x4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.drop_inner_tiled_unit_dims
    } : !transform.any_op
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] =  affine_map<() -> ()>

// CHECK-LABEL: func @drop_inner_tiled_unit_dims
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<1x1x4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<1x1x4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<1x1x4xf32>
//       CHECK:   %[[LHS_EXT:.+]] = vector.extract %[[LHS]][0, 0] : vector<4xf16> from vector<1x1x4xf16>
//       CHECK:   %[[RHS_EXT:.+]] = vector.extract %[[RHS]][0, 0] : vector<4xf16> from vector<1x1x4xf16>
//       CHECK:   %[[ACC_EXT:.+]] = vector.extract %[[ACC]][0, 0] : vector<4xf32> from vector<1x1x4xf32>
//       CHECK:   %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_EXT]], %[[RHS_EXT]]) outs(%[[ACC_EXT]])
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types = []
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>} : vector<4xf16>, vector<4xf16> into vector<4xf32>
//       CHECK:   vector.broadcast %[[MMA]] : vector<4xf32> to vector<1x1x4xf32>

// -----

#contraction_accesses = [
 affine_map<(i) -> (i)>,
 affine_map<(i) -> ()>,
 affine_map<(i) -> (i)>
]
func.func @drop_inner_tiled_unit_dims_no_kn(%lhs: vector<1x4xf16>, %rhs: vector<4xf16>, %acc: vector<1x4xf32>) -> vector<1x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } : vector<1x4xf16>, vector<4xf16> into vector<1x4xf32>
  return %0 : vector<1x4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.drop_inner_tiled_unit_dims
    } : !transform.any_op
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] =  affine_map<() -> ()>

// CHECK-LABEL: func @drop_inner_tiled_unit_dims_no_kn
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<1x4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<1x4xf32>
//       CHECK:   %[[LHS_EXT:.+]] = vector.extract %[[LHS]][0] : vector<4xf16> from vector<1x4xf16>
//       CHECK:   %[[ACC_EXT:.+]] = vector.extract %[[ACC]][0] : vector<4xf32> from vector<1x4xf32>
//       CHECK:   %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_EXT]], %[[RHS]]) outs(%[[ACC_EXT]])
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types = []
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>} : vector<4xf16>, vector<4xf16> into vector<4xf32>
//       CHECK:   vector.broadcast %[[MMA]] : vector<4xf32> to vector<1x4xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k, b) -> (i, k, b)>,
 affine_map<(i, j, k, b) -> (i, k)>,
 affine_map<(i, j, k, b) -> (k, b, j)>,
 affine_map<(i, j, k, b) -> (k, j)>,
 affine_map<(i, j, k, b) -> (i, j)>
]
func.func @drop_inner_tiled_scaled_mma_unit_dims(%lhs: vector<1x1x1x32xf4E2M1FN>, %lhsScale: vector<1x1x1xf8E8M0FNU>,
    %rhs: vector<1x1x1x32xf8E4M3FN>, %rhsScale: vector<1x1x1xf8E8M0FNU>,
    %acc: vector<1x1x4xf32>) -> vector<1x1x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %lhsScale, %rhs, %rhsScale) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>
  } : vector<1x1x1x32xf4E2M1FN>, vector<1x1x1xf8E8M0FNU>,
    vector<1x1x1x32xf8E4M3FN>, vector<1x1x1xf8E8M0FNU> into vector<1x1x4xf32>
  return %0 : vector<1x1x4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.drop_inner_tiled_unit_dims
    } : !transform.any_op
    transform.yield
  }
}

// CHECK: #[[$MAP:.+]] =  affine_map<() -> ()>

// CHECK-LABEL: func @drop_inner_tiled_scaled_mma_unit_dims
//       CHECK:   %[[MMA:.+]] = iree_codegen.inner_tiled
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]], #[[$MAP]], #[[$MAP]]], iterator_types = []
//  CHECK-SAME:     : vector<32xf4E2M1FN>, vector<1xf8E8M0FNU>, vector<32xf8E4M3FN>, vector<1xf8E8M0FNU> into vector<4xf32>
//       CHECK:   vector.broadcast %[[MMA]] : vector<4xf32> to vector<1x1x4xf32>
