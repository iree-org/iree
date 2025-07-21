// RUN: iree-opt %s --pass-pipeline='builtin.module(func.func(iree-gpu-distribute-inner-tiled-to-lanes, canonicalize, cse))' --split-input-file | FileCheck %s

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
module {
  func.func @matmul_16x16x16(%arg0: tensor<8x2x16x16xf16>, %arg1: tensor<8x2x16x16xf16>, %arg2: tensor<2x2x16x16xf32>) -> tensor<2x2x16x16xf32> {
    %empty = tensor.empty() : tensor<2x8x16x16xf16>
    %lhs_transpose = linalg.transpose ins(%arg0: tensor<8x2x16x16xf16>) outs(%empty: tensor<2x8x16x16xf16>) permutation = [1, 0, 2, 3]
    %mm = iree_codegen.inner_tiled ins(%lhs_transpose, %arg1) outs(%arg2) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
      permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]
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
//       CHECK:     iree_codegen.inner_tiled ins(%[[LHS_T]],
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
    %mm = iree_codegen.inner_tiled ins(%lhs_transpose, %arg1) outs(%arg2) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_I32_16x16x16_I8>,
      permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]
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
//       CHECK:     iree_codegen.inner_tiled ins(%[[LHS_T]]
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
    %mm = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
      permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1, 2>]
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
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]])
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
  func.func @col_major_matmul_32x32x8(%arg0: tensor<2x8x32x8xf16>, %arg1: tensor<8x2x32x8xf16>, %arg2: tensor<2x2x32x4x8xf32>) -> tensor<2x2x32x4x8xf32> {
    %mm = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16, col_major = true>,
      permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1, 2>]
    } : tensor<2x8x32x8xf16>, tensor<8x2x32x8xf16> into tensor<2x2x32x4x8xf32>
    return %mm : tensor<2x2x32x4x8xf32>
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @col_major_matmul_32x32x8
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<2x8x32x8xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<8x2x32x8xf16>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (64) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<2x2x32x4x8xf32>)
//   CHECK-DAG:     %[[ID:.+]]:3 = affine.delinearize_index %[[LANEID]] into (2, 32)
//   CHECK-DAG:     %[[IDY:.+]] = affine.linearize_index disjoint [%[[ID]]#1, %c0] by (2, 4)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0, %[[ID]]#2, %[[IDY]]] [2, 8, 1, 4]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0, %[[ID]]#2, %[[IDY]]] [8, 2, 1, 4]
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, 0, %[[ID]]#2, 0, %[[IDY]]] [2, 2, 1, 4, 4]
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]])
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16, col_major = true>
//  CHECK-SAME:       : tensor<2x8x1x4xf16>, tensor<8x2x1x4xf16> into tensor<2x2x1x4x4xf32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][0, 0, %[[ID]]#2, 0, %[[IDY]]] [2, 2, 1, 4, 4]
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
module {
  func.func @matmul_I32_32x32x8_I8(%arg0: tensor<2x8x32x8xi8>, %arg1: tensor<8x2x32x8xi8>, %arg2: tensor<2x2x4x8x32xi32>) -> tensor<2x2x4x8x32xi32> {
    %mm = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_I32_32x32x8_I8>,
      permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1, 2>]
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
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]])
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
  func.func @matmul_WMMAR3_16x16x16(%arg0: tensor<2x8x16x16xf16>, %arg1: tensor<8x2x16x16xf16>, %arg2: tensor<2x2x8x2x16xf32>) -> tensor<2x2x8x2x16xf32> {
    %mm = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>,
      permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1, 2>]
    } : tensor<2x8x16x16xf16>, tensor<8x2x16x16xf16> into tensor<2x2x8x2x16xf32>
    return %mm : tensor<2x2x8x2x16xf32>
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @matmul_WMMAR3_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<2x8x16x16xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<8x2x16x16xf16>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (32) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<2x2x8x2x16xf32>)
//   CHECK-DAG:     %[[ID_1:.+]]:2 = affine.delinearize_index %[[LANEID]] into (16)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0, %[[ID_1]]#1, 0] [2, 8, 1, 16]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0, %[[ID_1]]#1, 0] [8, 2, 1, 16]
//   CHECK-DAG:     %[[ID_2:.+]]:3 = affine.delinearize_index %[[LANEID]] into (2, 16)
//   Note: ID_2#1 and I_2#2 should not be delinearize outputs once we move to linearized indexing
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, 0, 0, %[[ID_2]]#1, %[[ID_2]]#2] [2, 2, 8, 1, 1]
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]])
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<WMMAR3_F32_16x16x16_F16>
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
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
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
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]])
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
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
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
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]])
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
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
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
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]])
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
func.func @distribute_WMMAR3_F16_16x16x16_F16(%lhs: tensor<16x16xf16>, %rhs: tensor<16x16xf16>, %acc: tensor<16x8x2xf16>) -> tensor<16x8x2xf16> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMAR3_F16_16x16x16_F16>
  } : tensor<16x16xf16>, tensor<16x16xf16> into tensor<16x8x2xf16>
  return %0 : tensor<16x8x2xf16>
}

// CHECK-LABEL: func @distribute_WMMAR3_F16_16x16x16_F16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<16x16xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<16x16xf16>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (32) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<16x8x2xf16>)
//   CHECK-DAG:     %[[ID:.+]]:2 = affine.delinearize_index %[[LANEID]] into (16)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][%[[ID]]#1, 0] [1, 16]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, %[[ID]]#1] [16, 1]
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, 0, %[[ID]]#1] [16, 1, 1]
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]])
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<WMMAR3_F16_16x16x16_F16>
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
  func.func @matmul_WMMAR3_i32_16x16x16_i8(%arg0: tensor<2x8x16x16xi8>, %arg1: tensor<8x2x16x16xi8>, %arg2: tensor<2x2x8x2x16xi32>) -> tensor<2x2x8x2x16xi32> {
    %mm = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<WMMAR3_I32_16x16x16_I8>,
      permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1, 2>]
    } : tensor<2x8x16x16xi8>, tensor<8x2x16x16xi8> into tensor<2x2x8x2x16xi32>
    return %mm : tensor<2x2x8x2x16xi32>
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @matmul_WMMAR3_i32_16x16x16_i8
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<2x8x16x16xi8>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<8x2x16x16xi8>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (32) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<2x2x8x2x16xi32>)
//   CHECK-DAG:     %[[ID:.+]]:2 = affine.delinearize_index %[[LANEID]] into (16)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0, %[[ID]]#1, 0] [2, 8, 1, 16]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0, %[[ID]]#1, 0] [8, 2, 1, 16]
//   CHECK-DAG:     %[[ID_ACC:.+]]:3 = affine.delinearize_index %[[LANEID]] into (2, 16)
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, 0, 0, %[[ID_ACC]]#1, %[[ID_ACC]]#2] [2, 2, 8, 1, 1]
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]])
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<WMMAR3_I32_16x16x16_I8>
//  CHECK-SAME:       : tensor<2x8x1x16xi8>, tensor<8x2x1x16xi8> into tensor<2x2x8x1x1xi32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][0, 0, 0, %[[ID_ACC]]#1, %[[ID_ACC]]#2] [2, 2, 8, 1, 1]
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @distribute_WMMAR4_F16_16x16x16_F16(%lhs: tensor<16x16xf16>, %rhs: tensor<16x16xf16>, %acc: tensor<16x16xf16>) -> tensor<16x16xf16> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<WMMAR4_F16_16x16x16_F16>
  } : tensor<16x16xf16>, tensor<16x16xf16> into tensor<16x16xf16>
  return %0 : tensor<16x16xf16>
}

// CHECK-LABEL: func @distribute_WMMAR4_F16_16x16x16_F16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<16x16xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<16x16xf16>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (32) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<16x16xf16>)
//   CHECK-DAG:     %[[ID:.+]]:3 = affine.delinearize_index %[[LANEID]] into (2, 16)
//   CHECK-DAG:     %[[IDY:.+]] = affine.linearize_index disjoint [%[[ID]]#1, %c0] by (2, 8)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][%[[ID]]#2, %[[IDY]]] [1, 8]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][%[[IDY]], %[[ID]]#2] [8, 1]
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][%[[IDY]], %[[ID]]#2] [8, 1]
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]])
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<WMMAR4_F16_16x16x16_F16>
//  CHECK-SAME:       : tensor<1x8xf16>, tensor<8x1xf16> into tensor<8x1xf16>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][%[[IDY]], %[[ID]]#2] [8, 1]
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
module {
  func.func @matmul_WMMAR4_i32_16x16x16_i8(%arg0: tensor<2x8x16x16xi8>, %arg1: tensor<8x2x16x16xi8>, %arg2: tensor<2x2x16x16xi32>) -> tensor<2x2x16x16xi32> {
    %mm = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2) {
      indexing_maps = #contraction_accesses,
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<WMMAR4_I32_16x16x16_I8>,
      permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]
    } : tensor<2x8x16x16xi8>, tensor<8x2x16x16xi8> into tensor<2x2x16x16xi32>
    return %mm : tensor<2x2x16x16xi32>
  }
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @matmul_WMMAR4_i32_16x16x16_i8
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<2x8x16x16xi8>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<8x2x16x16xi8>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (32) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<2x2x16x16xi32>)
//   CHECK-DAG:     %[[ID:.+]]:3 = affine.delinearize_index %[[LANEID]] into (2, 16)
//   CHECK-DAG:     %[[IDY:.+]] = affine.linearize_index disjoint [%[[ID]]#1, %c0] by (2, 8)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0, %[[ID]]#2, %[[IDY]]] [2, 8, 1, 8]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0, %[[ID]]#2, %[[IDY]]] [8, 2, 1, 8]
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, 0, %[[IDY]], %[[ID]]#2] [2, 2, 8, 1]
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]])
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<WMMAR4_I32_16x16x16_I8>
//  CHECK-SAME:       : tensor<2x8x1x8xi8>, tensor<8x2x1x8xi8> into tensor<2x2x8x1xi32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][0, 0, %[[IDY]], %[[ID]]#2] [2, 2, 8, 1]
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @data_tiled_1x1x1_tensor_multi_mma(%lhs: tensor<1x1x4x4x4xf32>, %rhs: tensor<1x1x4x16xf32>, %acc: tensor<1x1x4x16x4xf32>) -> tensor<1x1x4x16x4xf32>
      attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64>} {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
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
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]])
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
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
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
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]])
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
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
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
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[RHS_SLICE]]) outs(%[[ACC_SLICE]])
//  CHECK-SAME:       kind = #iree_gpu.data_tiled_mma_layout<intrinsic =  MFMA_F32_16x16x4_F32, subgroups_m = 2, subgroups_n = 2, intrinsics_k = 4>}
//  CHECK-SAME:       : tensor<1x1x1x1x1x1x4xf32>, tensor<1x1x1x1x1x4xf32> into tensor<1x1x1x1x1x1x4xf32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC_ARG]]
//  CHECK-SAME:       [0, 0, %[[ACC_IDS]]#1, %[[ACC_IDS]]#2, %[[ACC_IDS]]#3, %[[ACC_IDS]]#4, 0] [1, 1, 1, 1, 1, 1, 4] [1, 1, 1, 1, 1, 1, 1]
//       CHECK:   mapping = [#gpu.thread<linear_dim_0>]

// -----

#contraction_accesses = [
 affine_map<(i, j, k, b) -> (i, k, b)>,
 affine_map<(i, j, k, b) -> (i, k)>,
 affine_map<(i, j, k, b) -> (k, b, j)>,
 affine_map<(i, j, k, b) -> (k, j)>,
 affine_map<(i, j, k, b) -> (i, j)>
]

func.func @scaled_matmul_f32_16x16x128_b32_fp4_fp8(%lhs: tensor<3x5x1x16x4x32xf4E2M1FN>, %lhsScale: tensor<3x5x16x4xf8E8M0FNU>,
    %rhs: tensor<5x1x7x4x32x16xf8E4M3FN>, %rhsScale: tensor<5x7x4x16xf8E8M0FNU>,
    %acc: tensor<3x7x16x16xf32>) -> tensor<3x7x16x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %lhsScale, %rhs, %rhsScale) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.scaled_mma_layout<
      intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN,
      rhs_elem_type = f8E4M3FN,
      acc_elem_type = f32>
  } : tensor<3x5x1x16x4x32xf4E2M1FN>, tensor<3x5x16x4xf8E8M0FNU>,
    tensor<5x1x7x4x32x16xf8E4M3FN>, tensor<5x7x4x16xf8E8M0FNU>
    into tensor<3x7x16x16xf32>
  return %0 : tensor<3x7x16x16xf32>
}

// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
// CHECK-DAG: #[[$MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1)>
// CHECK-DAG: #[[$MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// CHECK-LABEL: func @scaled_matmul_f32_16x16x128_b32_fp4_fp8
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<3x5x1x16x4x32xf4E2M1FN>
//  CHECK-SAME:   %[[LHS_SCALE:[A-Za-z0-9]+]]: tensor<3x5x16x4xf8E8M0FNU>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<5x1x7x4x32x16xf8E4M3FN>
//  CHECK-SAME:   %[[RHS_SCALE:[A-Za-z0-9]+]]: tensor<5x7x4x16xf8E8M0FNU>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (64) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<3x7x16x16xf32>)
//   CHECK-DAG:     %[[ID:.+]]:3 = affine.delinearize_index %[[LANEID]] into (4, 16)
//   CHECK-DAG:     %[[IDY:.+]] = affine.linearize_index disjoint [%[[ID]]#1, %c0] by (4, 4)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0, 0, %[[ID]]#2, %[[ID]]#1, 0] [3, 5, 1, 1, 1, 32]
//   CHECK-DAG:     %[[LHS_SCALE_SLICE:.+]] = tensor.extract_slice %[[LHS_SCALE]][0, 0, %[[ID]]#2, %[[ID]]#1] [3, 5, 1, 1]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0, 0, %[[ID]]#1, 0, %[[ID]]#2] [5, 1, 7, 1, 32, 1]
//   CHECK-DAG:     %[[RHS_SCALE_SLICE:.+]] = tensor.extract_slice %[[RHS_SCALE]][0, 0, %[[ID]]#1, %[[ID]]#2] [5, 7, 1, 1]
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, 0, %[[IDY]], %[[ID]]#2] [3, 7, 4, 1]
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[LHS_SCALE_SLICE]], %[[RHS_SLICE]], %[[RHS_SCALE_SLICE]]) outs(%[[ACC_SLICE]])
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]], #[[$MAP3]], #[[$MAP4]]]
//  CHECK-SAME:       : tensor<3x5x1x1x1x32xf4E2M1FN>, tensor<3x5x1x1xf8E8M0FNU>, tensor<5x1x7x1x32x1xf8E4M3FN>, tensor<5x7x1x1xf8E8M0FNU> into tensor<3x7x4x1xf32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][0, 0, %[[IDY]], %[[ID]]#2] [3, 7, 4, 1]
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

// Note: thete tests don't check the affine maps and the like since it's assumed
// the above tests covered that code, which doesn't depend on the mma kind.

#contraction_accesses = [
 affine_map<(i, j, k, b) -> (i, k, b)>,
 affine_map<(i, j, k, b) -> (i, k)>,
 affine_map<(i, j, k, b) -> (k, b, j)>,
 affine_map<(i, j, k, b) -> (k, j)>,
 affine_map<(i, j, k, b) -> (i, j)>
]

func.func @scaled_matmul_trb_f32_16x16x128_b32_fp4_fp8(%lhs: tensor<3x5x4x16x4x32xf4E2M1FN>, %lhsScale: tensor<3x5x16x4xf8E8M0FNU>,
    %rhs: tensor<5x4x7x16x4x32xf8E4M3FN>, %rhsScale: tensor<5x7x16x4xf8E8M0FNU>,
    %acc: tensor<3x7x16x16xf32>) -> tensor<3x7x16x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %lhsScale, %rhs, %rhsScale) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.scaled_mma_layout<
      intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN,
      rhs_elem_type = f8E4M3FN,
      acc_elem_type = f32>,
    permutations = [array<i64: 0, 1, 2>, array<i64: 0, 1>,
      array<i64: 2, 0, 1>, array<i64: 1, 0>,
      array<i64: 0, 1>]
  } : tensor<3x5x4x16x4x32xf4E2M1FN>, tensor<3x5x16x4xf8E8M0FNU>,
    tensor<5x4x7x16x4x32xf8E4M3FN>, tensor<5x7x16x4xf8E8M0FNU>
    into tensor<3x7x16x16xf32>
  return %0 : tensor<3x7x16x16xf32>
}

// CHECK-LABEL: func @scaled_matmul_trb_f32_16x16x128_b32_fp4_fp8
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<3x5x4x16x4x32xf4E2M1FN>
//  CHECK-SAME:   %[[LHS_SCALE:[A-Za-z0-9]+]]: tensor<3x5x16x4xf8E8M0FNU>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<5x4x7x16x4x32xf8E4M3FN>
//  CHECK-SAME:   %[[RHS_SCALE:[A-Za-z0-9]+]]: tensor<5x7x16x4xf8E8M0FNU>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (64) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<3x7x16x16xf32>)
//   CHECK-DAG:     %[[ID:.+]]:3 = affine.delinearize_index %[[LANEID]] into (4, 16)
//   CHECK-DAG:     %[[IDY:.+]] = affine.linearize_index disjoint [%[[ID]]#1, %c0] by (4, 4)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0, 0, %[[ID]]#2, %[[ID]]#1, 0] [3, 5, 4, 1, 1, 32]
//   CHECK-DAG:     %[[LHS_SCALE_SLICE:.+]] = tensor.extract_slice %[[LHS_SCALE]][0, 0, %[[ID]]#2, %[[ID]]#1] [3, 5, 1, 1]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0, 0, %[[ID]]#2, %[[ID]]#1, 0] [5, 4, 7, 1, 1, 32]
//   CHECK-DAG:     %[[RHS_SCALE_SLICE:.+]] = tensor.extract_slice %[[RHS_SCALE]][0, 0, %[[ID]]#2, %[[ID]]#1] [5, 7, 1, 1]
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, 0, %[[IDY]], %[[ID]]#2] [3, 7, 4, 1]
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[LHS_SCALE_SLICE]], %[[RHS_SLICE]], %[[RHS_SCALE_SLICE]]) outs(%[[ACC_SLICE]])
//  CHECK-SAME:       : tensor<3x5x4x1x1x32xf4E2M1FN>, tensor<3x5x1x1xf8E8M0FNU>, tensor<5x4x7x1x1x32xf8E4M3FN>, tensor<5x7x1x1xf8E8M0FNU> into tensor<3x7x4x1xf32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][0, 0, %[[IDY]], %[[ID]]#2] [3, 7, 4, 1]
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#contraction_accesses = [
 affine_map<(i, j, k, b) -> (i, k, b)>,
 affine_map<(i, j, k, b) -> (i, k)>,
 affine_map<(i, j, k, b) -> (k, b, j)>,
 affine_map<(i, j, k, b) -> (k, j)>,
 affine_map<(i, j, k, b) -> (i, j)>
]

func.func @scaled_matmul_trb_f32_32x32x64_b32_fp4_fp8(%lhs: tensor<3x5x1x32x2x32xf4E2M1FN>, %lhsScale: tensor<3x5x32x2xf8E8M0FNU>,
    %rhs: tensor<5x1x7x32x2x32xf8E4M3FN>, %rhsScale: tensor<5x7x32x2xf8E8M0FNU>,
    %acc: tensor<3x7x4x8x32xf32>) -> tensor<3x7x4x8x32xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %lhsScale, %rhs, %rhsScale) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.scaled_mma_layout<
      intrinsic = MFMA_SCALE_F32_32x32x64_B32,
      lhs_elem_type = f4E2M1FN,
      rhs_elem_type = f8E4M3FN,
      acc_elem_type = f32>,
    permutations = [array<i64: 0, 1, 2>, array<i64: 0, 1>,
      array<i64: 2, 0, 1>, array<i64: 1, 0>,
      array<i64: 0, 1, 2>]
  } : tensor<3x5x1x32x2x32xf4E2M1FN>, tensor<3x5x32x2xf8E8M0FNU>,
    tensor<5x1x7x32x2x32xf8E4M3FN>, tensor<5x7x32x2xf8E8M0FNU>
    into tensor<3x7x4x8x32xf32>
  return %0 : tensor<3x7x4x8x32xf32>
}

// CHECK-LABEL: func @scaled_matmul_trb_f32_32x32x64_b32_fp4_fp8
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<3x5x1x32x2x32xf4E2M1FN>
//  CHECK-SAME:   %[[LHS_SCALE:[A-Za-z0-9]+]]: tensor<3x5x32x2xf8E8M0FNU>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<5x1x7x32x2x32xf8E4M3FN>
//  CHECK-SAME:   %[[RHS_SCALE:[A-Za-z0-9]+]]: tensor<5x7x32x2xf8E8M0FNU>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (64) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<3x7x4x8x32xf32>)
//   CHECK-DAG:     %[[ID:.+]]:3 = affine.delinearize_index %[[LANEID]] into (2, 32)
//   CHECK-DAG:     %[[IDY:.+]] = affine.linearize_index disjoint [%[[ID]]#1, %c0] by (2, 4)
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0, 0, %[[ID]]#2, %[[ID]]#1, 0] [3, 5, 1, 1, 1, 32]
//   CHECK-DAG:     %[[LHS_SCALE_SLICE:.+]] = tensor.extract_slice %[[LHS_SCALE]][0, 0, %[[ID]]#2, %[[ID]]#1] [3, 5, 1, 1]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0, 0, %[[ID]]#2, %[[ID]]#1, 0] [5, 1, 7, 1, 1, 32]
//   CHECK-DAG:     %[[RHS_SCALE_SLICE:.+]] = tensor.extract_slice %[[RHS_SCALE]][0, 0, %[[ID]]#2, %[[ID]]#1] [5, 7, 1, 1]
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, 0, 0, %[[IDY]], %[[ID]]#2] [3, 7, 4, 4, 1]
//       CHECK:     %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS_SLICE]], %[[LHS_SCALE_SLICE]], %[[RHS_SLICE]], %[[RHS_SCALE_SLICE]]) outs(%[[ACC_SLICE]])
//  CHECK-SAME:       : tensor<3x5x1x1x1x32xf4E2M1FN>, tensor<3x5x1x1xf8E8M0FNU>, tensor<5x1x7x1x1x32xf8E4M3FN>, tensor<5x7x1x1xf8E8M0FNU> into tensor<3x7x4x4x1xf32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][0, 0, 0, %[[IDY]], %[[ID]]#2] [3, 7, 4, 4, 1]
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @distributed_matmul(%arg0: vector<2x8x1x4xf16>, %arg1: vector<8x2x1x4xf16>, %arg2: vector<2x2x4x1xf32>) -> vector<2x2x4x1xf32> {
    %0 = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2) {indexing_maps = [#map, #map1, #map2],
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>} : vector<2x8x1x4xf16>, vector<8x2x1x4xf16> into vector<2x2x4x1xf32>
    return %0 : vector<2x2x4x1xf32>
  }
}

// Verify that already vectorized (assumed distributed) mma ops are pass through.
// CHECK-LABEL: func @distributed_matmul
//       CHECK:   iree_codegen.inner_tiled {{.*}} : vector<2x8x1x4xf16>, vector<8x2x1x4xf16> into vector<2x2x4x1xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @fuse_producer_slice(%arg1 : tensor<4x2x16x16xbf16>, %arg2 : tensor<1x2x16x16xbf16>, %arg3 : tensor<4x1x16x16xf32>) -> tensor<4x1x16x16xf32> {
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = linalg.fill ins(%cst_0 : f32) outs(%arg3 : tensor<4x1x16x16xf32>) -> tensor<4x1x16x16xf32>
  %result = iree_codegen.inner_tiled ins(%arg1, %arg2) outs(%0) {indexing_maps = [#map, #map1, #map2],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_BF16>
    } : tensor<4x2x16x16xbf16>, tensor<1x2x16x16xbf16> into tensor<4x1x16x16xf32>
  return %result : tensor<4x1x16x16xf32>
}

// CHECK-LABEL: func @fuse_producer_slice
// CHECK      :   scf.forall (%[[LANEID:.+]]) in (64) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<4x1x16x16xf32>)
// CHECK      :     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]]
// CHECK      :     %[[FILL:.+]] = linalg.fill ins(%cst : f32) outs(%[[ACC_SLICE]] : tensor<4x1x4x1xf32>) -> tensor<4x1x4x1xf32>
// CHECK      :     iree_codegen.inner_tiled
// CHECK-SAME :     outs(%[[FILL]])
// CHECK      :     mapping = [#iree_gpu.lane_id<0>]
