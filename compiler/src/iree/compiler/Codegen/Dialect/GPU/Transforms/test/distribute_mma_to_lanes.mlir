// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-gpu-distribute-mma-to-lanes, canonicalize, cse))' --split-input-file | FileCheck %s

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
module {
  func.func @matmul_16x16x16(%arg0: tensor<8x2x16x16xf16>, %arg1: tensor<8x2x16x16xf16>, %arg2: tensor<2x2x16x16xf32>) -> tensor<2x2x16x16xf32> {
    %empty = tensor.empty() : tensor<2x8x16x16xf16>
    %lhs_transpose = linalg.transpose ins(%arg0: tensor<8x2x16x16xf16>) outs(%empty: tensor<2x8x16x16xf16>) permutation = [1, 0, 2, 3]
    %mm = iree_gpu.multi_mma %lhs_transpose, %arg1, %arg2 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
      rhs_permutation = array<i64: 1, 0>
    } : tensor<2x8x16x16xf16>, tensor<8x2x16x16xf16> into tensor<2x2x16x16xf32>
    return %mm : tensor<2x2x16x16xf32>
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @matmul_16x16x16
//       CHECK:   scf.forall
//       CHECK:     %[[LHS_T:.+]] = linalg.transpose ins({{.*}}: tensor<2x8x1x4xf16>)
//       CHECK:     iree_gpu.multi_mma %[[LHS_T]]
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
//  CHECK-SAME:       : tensor<2x8x1x4xf16>, tensor<8x2x1x4xf16> into tensor<2x2x4x1xf32>
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
module {
  func.func @matmul_I32_16x16x16_I8(%arg0: tensor<8x2x16x16xi8>, %arg1: tensor<8x2x16x16xi8>, %arg2: tensor<2x2x16x16xi32>) -> tensor<2x2x16x16xi32> {
    %empty = tensor.empty() : tensor<2x8x16x16xi8>
    %lhs_transpose = linalg.transpose ins(%arg0: tensor<8x2x16x16xi8>) outs(%empty: tensor<2x8x16x16xi8>) permutation = [1, 0, 2, 3]
    %mm = iree_gpu.multi_mma %lhs_transpose, %arg1, %arg2 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_I32_16x16x16_I8>,
      rhs_permutation = array<i64: 1, 0>
    } : tensor<2x8x16x16xi8>, tensor<8x2x16x16xi8> into tensor<2x2x16x16xi32>
    return %mm : tensor<2x2x16x16xi32>
  }
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @matmul_I32_16x16x16_I8
//       CHECK:   scf.forall
//       CHECK:     %[[LHS_T:.+]] = linalg.transpose ins({{.*}}: tensor<2x8x1x4xi8>)
//       CHECK:     iree_gpu.multi_mma %[[LHS_T]]
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_I32_16x16x16_I8>
//  CHECK-SAME:       : tensor<2x8x1x4xi8>, tensor<8x2x1x4xi8> into tensor<2x2x4x1xi32>
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
module {
  func.func @matmul_32x32x8(%arg0: tensor<2x8x32x8xf16>, %arg1: tensor<8x2x32x8xf16>, %arg2: tensor<2x2x4x8x32xf32>) -> tensor<2x2x4x8x32xf32> {
    %mm = iree_gpu.multi_mma %arg0, %arg1, %arg2 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
      rhs_permutation = array<i64: 1, 0>
    } : tensor<2x8x32x8xf16>, tensor<8x2x32x8xf16> into tensor<2x2x4x8x32xf32>
    return %mm : tensor<2x2x4x8x32xf32>
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @matmul_32x32x8
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<2x8x32x8xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<8x2x32x8xf16>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (64) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<2x2x4x8x32xf32>)
//   CHECK-DAG:     %[[ID:.+]]:3 = affine.delinearize_index %[[LANEID]] into (2, 32)
//   CHECK-DAG:     %[[IDY:.+]] = affine.linearize_index disjoint [%[[ID]]#1, %c0] by (2, 4)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0, %[[ID]]#2, %[[IDY]]] [2, 8, 1, 4]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0, %[[ID]]#2, %[[IDY]]] [8, 2, 1, 4]
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, 0, 0, %[[IDY]], %[[ID]]#2] [2, 2, 4, 4, 1]
//       CHECK:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]]
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>
//  CHECK-SAME:       : tensor<2x8x1x4xf16>, tensor<8x2x1x4xf16> into tensor<2x2x4x4x1xf32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][0, 0, 0, %[[IDY]], %[[ID]]#2] [2, 2, 4, 4, 1]
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
module {
  func.func @matmul_I32_32x32x8_I8(%arg0: tensor<2x8x32x8xi8>, %arg1: tensor<8x2x32x8xi8>, %arg2: tensor<2x2x4x8x32xi32>) -> tensor<2x2x4x8x32xi32> {
    %mm = iree_gpu.multi_mma %arg0, %arg1, %arg2 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_I32_32x32x8_I8>,
      rhs_permutation = array<i64: 1, 0>
    } : tensor<2x8x32x8xi8>, tensor<8x2x32x8xi8> into tensor<2x2x4x8x32xi32>
    return %mm : tensor<2x2x4x8x32xi32>
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @matmul_I32_32x32x8_I8
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<2x8x32x8xi8>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<8x2x32x8xi8>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (64) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<2x2x4x8x32xi32>)
//   CHECK-DAG:     %[[ID:.+]]:3 = affine.delinearize_index %[[LANEID]] into (2, 32)
//   CHECK-DAG:     %[[IDY:.+]] = affine.linearize_index disjoint [%[[ID]]#1, %c0] by (2, 4)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0, %[[ID]]#2, %[[IDY]]] [2, 8, 1, 4]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0, %[[ID]]#2, %[[IDY]]] [8, 2, 1, 4]
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, 0, 0, %[[IDY]], %[[ID]]#2] [2, 2, 4, 4, 1]
//       CHECK:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]]
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_I32_32x32x8_I8>
//  CHECK-SAME:       : tensor<2x8x1x4xi8>, tensor<8x2x1x4xi8> into tensor<2x2x4x4x1xi32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][0, 0, 0, %[[IDY]], %[[ID]]#2] [2, 2, 4, 4, 1]
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
module {
  func.func @matmul_wmma_16x16x16(%arg0: tensor<2x8x16x16xf16>, %arg1: tensor<8x2x16x16xf16>, %arg2: tensor<2x2x8x2x16xf32>) -> tensor<2x2x8x2x16xf32> {
    %mm = iree_gpu.multi_mma %arg0, %arg1, %arg2 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<WMMA_F32_16x16x16_F16>,
      rhs_permutation = array<i64: 1, 0>
    } : tensor<2x8x16x16xf16>, tensor<8x2x16x16xf16> into tensor<2x2x8x2x16xf32>
    return %mm : tensor<2x2x8x2x16xf32>
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @matmul_wmma_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<2x8x16x16xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<8x2x16x16xf16>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (32) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<2x2x8x2x16xf32>)
//   CHECK-DAG:     %[[ID_1:.+]]:2 = affine.delinearize_index %[[LANEID]] into (16)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0, %[[ID_1]]#1, 0] [2, 8, 1, 16]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0, %[[ID_1]]#1, 0] [8, 2, 1, 16]
//   CHECK-DAG:     %[[ID_2:.+]]:3 = affine.delinearize_index %[[LANEID]] into (2, 16)
//   Note: ID_2#1 and I_2#2 should not be delinearize outputs once we move to linearized indexing
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, 0, 0, %[[ID_2]]#1, %[[ID_2]]#2] [2, 2, 8, 1, 1]
//       CHECK:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]]
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<WMMA_F32_16x16x16_F16>
//  CHECK-SAME:       : tensor<2x8x1x16xf16>, tensor<8x2x1x16xf16> into tensor<2x2x8x1x1xf32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][0, 0, 0, %[[ID_2]]#1, %[[ID_2]]#2] [2, 2, 8, 1, 1]
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @distribute_MFMA_F32_16x16x4_F32(%lhs: tensor<16x4xf32>, %rhs: tensor<4x16xf32>, %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>
  } : tensor<16x4xf32>, tensor<4x16xf32> into tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// CHECK-LABEL: func @distribute_MFMA_F32_16x16x4_F32
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<16x4xf32>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<4x16xf32>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (64) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<16x16xf32>)
//   CHECK-DAG:     %[[ID:.+]]:3 = affine.delinearize_index %[[LANEID]] into (4, 16)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][%[[ID]]#2, %[[ID]]#1] [1, 1]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][%[[ID]]#1, %[[ID]]#2] [1, 1]
//   CHECK-DAG:     %[[IDZ:.+]] = affine.linearize_index disjoint [%[[ID]]#1, %c0] by (4, 4)
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][%[[IDZ]], %[[ID]]#2] [4, 1]
//       CHECK:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>
//  CHECK-SAME:       : tensor<1x1xf32>, tensor<1x1xf32> into tensor<4x1xf32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][%[[IDZ]], %[[ID]]#2] [4, 1]
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @distribute_F32_16x16x32_F8E4M3FNUZ(%lhs: tensor<16x32xf8E4M3FNUZ>, %rhs: tensor<32x16xf8E4M3FNUZ>, %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
  } : tensor<16x32xf8E4M3FNUZ>, tensor<32x16xf8E4M3FNUZ> into tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// CHECK-LABEL: func @distribute_F32_16x16x32_F8E4M3FNUZ
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<16x32xf8E4M3FNUZ>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<32x16xf8E4M3FNUZ>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (64) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<16x16xf32>)
//   CHECK-DAG:     %[[ID:.+]]:3 = affine.delinearize_index %[[LANEID]] into (4, 16)
//   CHECK-DAG:     %[[IDY:.+]] = affine.linearize_index disjoint [%[[ID]]#1, %c0] by (4, 8)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][%[[ID]]#2, %[[IDY]]] [1, 8]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][%[[IDY]], %[[ID]]#2] [8, 1]
//   CHECK-DAG:     %[[IDZ:.+]] = affine.linearize_index disjoint [%[[ID]]#1, %c0] by (4, 4)
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][%[[IDZ]], %[[ID]]#2] [4, 1]
//       CHECK:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F8E4M3FNUZ>
//  CHECK-SAME:       : tensor<1x8xf8E4M3FNUZ>, tensor<8x1xf8E4M3FNUZ> into tensor<4x1xf32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][%[[IDZ]], %[[ID]]#2] [4, 1]
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @distribute_I32_32x32x16_I8(%lhs: tensor<32x16xi8>, %rhs: tensor<16x32xi8>, %acc: tensor<4x8x32xi32>) -> tensor<4x8x32xi32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>
  } : tensor<32x16xi8>, tensor<16x32xi8> into tensor<4x8x32xi32>
  return %0 : tensor<4x8x32xi32>
}

// CHECK-LABEL: func @distribute_I32_32x32x16_I8
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<32x16xi8>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<16x32xi8>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (64) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<4x8x32xi32>)
//   CHECK-DAG:     %[[ID:.+]]:3 = affine.delinearize_index %[[LANEID]] into (2, 32)
//   CHECK-DAG:     %[[IDY:.+]] = affine.linearize_index disjoint [%[[ID]]#1, %c0] by (2, 8)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][%[[ID]]#2, %[[IDY]]] [1, 8]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][%[[IDY]], %[[ID]]#2] [8, 1]
//   CHECK-DAG:     %[[IDZ:.+]] = affine.linearize_index disjoint [%[[ID]]#1, %c0] by (2, 4)
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, %[[IDZ]], %[[ID]]#2] [4, 4, 1]
//       CHECK:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_I32_32x32x16_I8>
//  CHECK-SAME:       : tensor<1x8xi8>, tensor<8x1xi8> into tensor<4x4x1xi32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][0, %[[IDZ]], %[[ID]]#2] [4, 4, 1]
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @distribute_WMMA_F16_16x16x16_F16(%lhs: tensor<16x16xf16>, %rhs: tensor<16x16xf16>, %acc: tensor<8x2x16xf16>) -> tensor<8x2x16xf16> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F16>
  } : tensor<16x16xf16>, tensor<16x16xf16> into tensor<8x2x16xf16>
  return %0 : tensor<8x2x16xf16>
}

// CHECK-LABEL: func @distribute_WMMA_F16_16x16x16_F16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<16x16xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<16x16xf16>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (32) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<8x2x16xf16>)
//   CHECK-DAG:     %[[ID:.+]]:2 = affine.delinearize_index %[[LANEID]] into (16)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][%[[ID]]#1, 0] [1, 16]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, %[[ID]]#1] [16, 1]
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, 0, %[[ID]]#1] [16, 1, 1]
//       CHECK:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<WMMA_F16_16x16x16_F16>
//  CHECK-SAME:       : tensor<1x16xf16>, tensor<16x1xf16> into tensor<16x1x1xf16>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][0, 0, %[[ID]]#1] [16, 1, 1]
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
module {
  func.func @matmul_wmma_i32_16x16x16_i8(%arg0: tensor<2x8x16x16xi8>, %arg1: tensor<8x2x16x16xi8>, %arg2: tensor<2x2x8x2x16xi32>) -> tensor<2x2x8x2x16xi32> {
    %mm = iree_gpu.multi_mma %arg0, %arg1, %arg2 {
      indexing_maps = #contraction_accesses,
      iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<WMMA_I32_16x16x16_I8>,
      rhs_permutation = array<i64: 1, 0>
    } : tensor<2x8x16x16xi8>, tensor<8x2x16x16xi8> into tensor<2x2x8x2x16xi32>
    return %mm : tensor<2x2x8x2x16xi32>
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @matmul_wmma_i32_16x16x16_i8
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<2x8x16x16xi8>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<8x2x16x16xi8>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (32) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<2x2x8x2x16xi32>)
//   CHECK-DAG:     %[[ID:.+]]:2 = affine.delinearize_index %[[LANEID]] into (16)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0, %[[ID]]#1, 0] [2, 8, 1, 16]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0, %[[ID]]#1, 0] [8, 2, 1, 16]
//   CHECK-DAG:     %[[ID_ACC:.+]]:3 = affine.delinearize_index %[[LANEID]] into (2, 16)
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, 0, 0, %[[ID_ACC]]#1, %[[ID_ACC]]#2] [2, 2, 8, 1, 1]
//       CHECK:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]]
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<WMMA_I32_16x16x16_I8>
//  CHECK-SAME:       : tensor<2x8x1x16xi8>, tensor<8x2x1x16xi8> into tensor<2x2x8x1x1xi32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][0, 0, 0, %[[ID_ACC]]#1, %[[ID_ACC]]#2] [2, 2, 8, 1, 1]
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @data_tiled_1x1x1_tensor_multi_mma(%lhs: tensor<1x1x4x4x4xf32>, %rhs: tensor<1x1x4x16xf32>, %acc: tensor<1x1x4x16x4xf32>) -> tensor<1x1x4x16x4xf32>
      attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>} {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32>
  } : tensor<1x1x4x4x4xf32>, tensor<1x1x4x16xf32> into tensor<1x1x4x16x4xf32>
  return %0 : tensor<1x1x4x16x4xf32>
}

// CHECK-LABEL: func @data_tiled_1x1x1_tensor_multi_mma
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]
//       CHECK:   scf.forall (%[[THREAD_ID:.+]]) in (64) shared_outs(%[[ACC_ARG:.+]] = %[[ACC]]) -> (tensor<1x1x4x16x4xf32>)
//   CHECK-DAG:     %[[LHS_IN_IDS:.+]]:4 = affine.delinearize_index %[[THREAD_ID]] into (4, 4, 4)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0, %[[LHS_IN_IDS]]#1, %[[LHS_IN_IDS]]#2, %[[LHS_IN_IDS]]#3] [1, 1, 1, 1, 1] [1, 1, 1, 1, 1]
//   CHECK-DAG:     %[[IN_IDS:.+]]:3 = affine.delinearize_index %[[THREAD_ID]] into (4, 16)
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0, %[[IN_IDS]]#1, %[[IN_IDS]]#2] [1, 1, 1, 1] [1, 1, 1, 1]
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC_ARG]]
//  CHECK-SAME:       [0, 0, %[[IN_IDS]]#1, %[[IN_IDS]]#2, 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1]
//       CHECK:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]]
//  CHECK-SAME:       kind = #iree_gpu.data_tiled_mma_layout<intrinsic =  MFMA_F32_16x16x4_F32>
//  CHECK-SAME:       : tensor<1x1x1x1x1xf32>, tensor<1x1x1x1xf32> into tensor<1x1x1x1x4xf32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC_ARG]]
//  CHECK-SAME:       [0, 0, %[[IN_IDS]]#1, %[[IN_IDS]]#2, 0] [1, 1, 1, 1, 4] [1, 1, 1, 1, 1]
//       CHECK:   mapping = [#gpu.thread<linear_dim_0>]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @data_tiled_2x2x4_tensor_multi_mma_unrolled(%lhs: tensor<1x1x2x4x4x4x4xf32>, %rhs: tensor<1x1x2x4x16x4xf32>, %acc: tensor<1x1x2x2x4x16x4xf32>) -> tensor<1x1x2x2x4x16x4xf32>
      attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>} {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, intrinsics_m = 2, intrinsics_n = 2, intrinsics_k = 4>
  } : tensor<1x1x2x4x4x4x4xf32>, tensor<1x1x2x4x16x4xf32> into tensor<1x1x2x2x4x16x4xf32>
  return %0 : tensor<1x1x2x2x4x16x4xf32>
}

// CHECK-LABEL: func @data_tiled_2x2x4_tensor_multi_mma_unrolled
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]
//       CHECK:   scf.forall (%[[THREAD_ID:.+]]) in (64) shared_outs(%[[ACC_ARG:.+]] = %[[ACC]]) -> (tensor<1x1x2x2x4x16x4xf32>)
//   CHECK-DAG:     %[[LHS_IN_IDS:.+]]:4 = affine.delinearize_index %[[THREAD_ID]] into (4, 4, 4)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]]
//  CHECK-SAME:       [0, 0, 0, %[[LHS_IN_IDS]]#1, %[[LHS_IN_IDS]]#2, %[[LHS_IN_IDS]]#3, 0] [1, 1, 2, 1, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1]
//   CHECK-DAG:     %[[IN_IDS:.+]]:3 = affine.delinearize_index %[[THREAD_ID]] into (4, 16)
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]]
//  CHECK-SAME:       [0, 0, 0, %[[IN_IDS]]#1, %[[IN_IDS]]#2, 0] [1, 1, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1]
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC_ARG]]
//  CHECK-SAME:       [0, 0, 0, 0, %[[IN_IDS]]#1, %[[IN_IDS]]#2, 0] [1, 1, 2, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1]
//       CHECK:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]]
//  CHECK-SAME:       kind = #iree_gpu.data_tiled_mma_layout<intrinsic =  MFMA_F32_16x16x4_F32, intrinsics_m = 2, intrinsics_n = 2, intrinsics_k = 4>
//  CHECK-SAME:       : tensor<1x1x2x1x1x1x4xf32>, tensor<1x1x2x1x1x4xf32> into tensor<1x1x2x2x1x1x4xf32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC_ARG]]
//  CHECK-SAME:       [0, 0, 0, 0, %[[IN_IDS]]#1, %[[IN_IDS]]#2, 0] [1, 1, 2, 2, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1]
//       CHECK:   mapping = [#gpu.thread<linear_dim_0>]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @data_tiled_2x2x4_tensor_multi_mma_unrolled_to_subgroups(%lhs: tensor<1x1x2x4x4x4x4xf32>, %rhs: tensor<1x1x2x4x16x4xf32>, %acc: tensor<1x1x2x2x4x16x4xf32>) -> tensor<1x1x2x2x4x16x4xf32>
      attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [256, 1, 1] subgroup_size = 64>} {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, subgroups_m = 2, subgroups_n = 2, intrinsics_k = 4>
  } : tensor<1x1x2x4x4x4x4xf32>, tensor<1x1x2x4x16x4xf32> into tensor<1x1x2x2x4x16x4xf32>
  return %0 : tensor<1x1x2x2x4x16x4xf32>
}

// CHECK-LABEL: func @data_tiled_2x2x4_tensor_multi_mma_unrolled_to_subgroups
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]
//  CHECK-SAME:   %[[ACC:[A-Za-z0-9]+]]
//       CHECK:   scf.forall (%[[THREAD_ID:.+]]) in (256) shared_outs(%[[ACC_ARG:.+]] = %[[ACC]]) -> (tensor<1x1x2x2x4x16x4xf32>)
//   CHECK-DAG:     %[[LHS_IN_IDS:.+]]:5 = affine.delinearize_index %[[THREAD_ID]] into (2, 4, 4, 4)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]]
//  CHECK-SAME:       [0, 0, %[[LHS_IN_IDS]]#1, %[[LHS_IN_IDS]]#2, %[[LHS_IN_IDS]]#3, %[[LHS_IN_IDS]]#4, 0] [1, 1, 1, 1, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1]
//   CHECK-DAG:     %[[IN_IDS:.+]]:4 = affine.delinearize_index %[[THREAD_ID]] into (2, 4, 16)
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]]
//  CHECK-SAME:       [0, 0, %[[IN_IDS]]#1, %[[IN_IDS]]#2, %[[IN_IDS]]#3, 0] [1, 1, 1, 1, 1, 4] [1, 1, 1, 1, 1, 1]
//   CHECK-DAG:     %[[ACC_IDS:.+]]:5 = affine.delinearize_index %[[THREAD_ID]] into (2, 2, 4, 16)
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC_ARG]]
//  CHECK-SAME:       [0, 0, %[[ACC_IDS]]#1, %[[ACC_IDS]]#2, %[[ACC_IDS]]#3, %[[ACC_IDS]]#4, 0] [1, 1, 1, 1, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1]
//       CHECK:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]]
//  CHECK-SAME:       kind = #iree_gpu.data_tiled_mma_layout<intrinsic =  MFMA_F32_16x16x4_F32, subgroups_m = 2, subgroups_n = 2, intrinsics_k = 4>}
//  CHECK-SAME:       : tensor<1x1x1x1x1x1x4xf32>, tensor<1x1x1x1x1x4xf32> into tensor<1x1x1x1x1x1x4xf32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC_ARG]]
//  CHECK-SAME:       [0, 0, %[[ACC_IDS]]#1, %[[ACC_IDS]]#2, %[[ACC_IDS]]#3, %[[ACC_IDS]]#4, 0] [1, 1, 1, 1, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1]
//       CHECK:   mapping = [#gpu.thread<linear_dim_0>]
