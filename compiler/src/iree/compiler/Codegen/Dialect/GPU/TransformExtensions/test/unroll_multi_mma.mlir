// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file | FileCheck %s

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @unroll_multi_mma_order(%lhs: vector<2x2x4xf16>, %rhs: vector<2x2x4xf16>, %acc: vector<2x2x4xf32>) -> vector<2x2x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<2x2x4xf16>, vector<2x2x4xf16> into vector<2x2x4xf32>
  return %0 : vector<2x2x4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.unroll_multi_mma
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @unroll_multi_mma_order
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: vector<2x2x4xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: vector<2x2x4xf16>
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]: vector<2x2x4xf32>

//       CHECK:   %[[ACC_DIST:.+]]:4 = util.hoistable_conversion "unroll_acc_distribute" inverts("unroll_acc_reassemble")
//  CHECK-SAME:     (%[[ACC_B:.+]]: vector<2x2x4xf32> = %[[ACC]])
//       CHECK:     vector.extract_strided_slice %[[ACC_B]] {offsets = [0, 0]
//       CHECK:     vector.extract_strided_slice %[[ACC_B]] {offsets = [0, 1]
//       CHECK:     vector.extract_strided_slice %[[ACC_B]] {offsets = [1, 0]
//       CHECK:     vector.extract_strided_slice %[[ACC_B]] {offsets = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [0, 0]
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [0, 0]
//       CHECK:   %[[MMA0_K0:.+]] = iree_codegen.inner_tiled ins(%{{.*}}, %{{.*}}) outs(%[[ACC_DIST]]#0)
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [0, 0]
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [0, 1]
//       CHECK:   %[[MMA1_K0:.+]] = iree_codegen.inner_tiled ins(%{{.*}}, %{{.*}}) outs(%[[ACC_DIST]]#1)
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [1, 0]
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [0, 0]
//       CHECK:   %[[MMA2_K0:.+]] = iree_codegen.inner_tiled ins(%{{.*}}, %{{.*}}) outs(%[[ACC_DIST]]#2)
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [1, 0]
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [0, 1]
//       CHECK:   %[[MMA3_K0:.+]] = iree_codegen.inner_tiled ins(%{{.*}}, %{{.*}}) outs(%[[ACC_DIST]]#3)
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [0, 1]
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [1, 0]
//       CHECK:   %[[MMA0:.+]] = iree_codegen.inner_tiled ins(%{{.*}}, %{{.*}}) outs(%[[MMA0_K0]])
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [0, 1]
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [1, 1]
//       CHECK:   %[[MMA1:.+]] = iree_codegen.inner_tiled ins(%{{.*}}, %{{.*}}) outs(%[[MMA1_K0]])
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [1, 0]
//       CHECK:   %[[MMA2:.+]] = iree_codegen.inner_tiled ins(%{{.*}}, %{{.*}}) outs(%[[MMA2_K0]])
//       CHECK:   vector.extract_strided_slice %[[LHS]] {offsets = [1, 1]
//       CHECK:   vector.extract_strided_slice %[[RHS]] {offsets = [1, 1]
//       CHECK:   %[[MMA3:.+]] = iree_codegen.inner_tiled ins(%{{.*}}, %{{.*}}) outs(%[[MMA3_K0]])
//       CHECK:   %[[RES:.+]] = util.hoistable_conversion "unroll_acc_reassemble" inverts("unroll_acc_distribute")
//       CHECK:     vector.insert_strided_slice %{{.+}}, %{{.+}} {offsets = [0, 0, 0]
//       CHECK:     vector.insert_strided_slice %{{.+}}, %{{.+}} {offsets = [0, 1, 0]
//       CHECK:     vector.insert_strided_slice %{{.+}}, %{{.+}} {offsets = [1, 0, 0]
//       CHECK:     vector.insert_strided_slice %{{.+}}, %{{.+}} {offsets = [1, 1, 0]
//       CHECK:   return %[[RES]]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @unroll_multi_mma_count(%lhs: vector<2x3x4xf16>, %rhs: vector<3x5x4xf16>, %acc: vector<2x5x4xf32>) -> vector<2x5x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>
  return %0 : vector<2x5x4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.unroll_multi_mma
    } : !transform.any_op
    transform.yield
  }
}

//   CHECK-LABEL: func @unroll_multi_mma_count
// CHECK-COUNT-30:   {{.+}} = iree_codegen.inner_tiled {{.*}} : vector<1x1x4xf16>, vector<1x1x4xf16> into vector<1x1x4xf32>
// CHECK-COUNT-10:   vector.insert_strided_slice {{.*}} : vector<1x1x4xf32> into vector<2x5x4xf32>

// -----

// Test unrolling scaled MFMA where the scales repeat every 64 elements (hence
// 2 x 32).
#contraction_accesses = [
 affine_map<(i, j, k, b) -> (i, k, b)>,
 affine_map<(i, j, k, b) -> (k, b, j)>,
 affine_map<(i, j, k, b) -> (i, k)>,
 affine_map<(i, j, k, b) -> (k, j)>,
 affine_map<(i, j, k, b) -> (i, j)>
]
func.func @unroll_scaled_multi_mma(%lhs: vector<1x2x2x32xf4E2M1FN>, %rhs: vector<2x2x1x32xf8E4M3FN>, %lhsScale: vector<1x2x1xf8E8M0FNU>, %rhsScale: vector<2x1x1xf8E8M0FNU>,
    %acc: vector<1x1x4xf32>) -> vector<1x1x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs, %lhsScale, %rhsScale) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<1x2x2x32xf4E2M1FN>, vector<2x2x1x32xf8E4M3FN>, vector<1x2x1xf8E8M0FNU>, vector<2x1x1xf8E8M0FNU> into vector<1x1x4xf32>
  return %0 : vector<1x1x4xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.unroll_multi_mma
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @unroll_scaled_multi_mma
//  CHECK-SAME:   %[[LHS_SCALE:[A-Za-z0-9]+]]: vector<1x2x1xf8E8M0FNU>
// CHECK-COUNT-2: vector.extract_strided_slice %[[LHS_SCALE]] {offsets = [0, 0]
// CHECK-NOT: vector.extract_strided_slice %[[LHS_SCALE]] {offsets = [0, 0]
// CHECK-COUNT-2: vector.extract_strided_slice %[[LHS_SCALE]] {offsets = [0, 1]
// CHECK-NOT: vector.extract_strided_slice %[[LHS_SCALE]]
