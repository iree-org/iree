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

// CHECK-DAG: #[[$XMAP:.+]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK-DAG: #[[$YMAP:.+]] = affine_map<(d0) -> ((d0 floordiv 32) * 4 - ((d0 floordiv 32) floordiv 2) * 8)>
// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @matmul_32x32x8
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<2x8x32x8xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<8x2x32x8xf16>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (64) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<2x2x4x8x32xf32>)
//   CHECK-DAG:     %[[IDX:.+]] = affine.apply #[[$XMAP]](%[[LANEID]])
//   CHECK-DAG:     %[[IDY:.+]] = affine.apply #[[$YMAP]](%[[LANEID]])
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0, %[[IDX]], %[[IDY]]] [2, 8, 1, 4]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0, %[[IDX]], %[[IDY]]] [8, 2, 1, 4]
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, 0, 0, %[[IDY]], %[[IDX]]] [2, 2, 4, 4, 1]
//       CHECK:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]]
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>
//  CHECK-SAME:       : tensor<2x8x1x4xf16>, tensor<8x2x1x4xf16> into tensor<2x2x4x4x1xf32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][0, 0, 0, %[[IDY]], %[[IDX]]] [2, 2, 4, 4, 1]
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

// CHECK-DAG: #[[$XMAP:.+]] = affine_map<(d0) -> (d0 mod 16)>
// CHECK-DAG: #[[$YMAP:.+]] = affine_map<(d0) -> ((d0 floordiv 16) mod 2)>
// CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @matmul_wmma_16x16x16
//  CHECK-SAME:   %[[LHS:[A-Za-z0-9]+]]: tensor<2x8x16x16xf16>
//  CHECK-SAME:   %[[RHS:[A-Za-z0-9]+]]: tensor<8x2x16x16xf16>
//       CHECK:   scf.forall (%[[LANEID:.+]]) in (32) shared_outs(%[[ACC:.+]] = {{.*}}) -> (tensor<2x2x8x2x16xf32>)
//   CHECK-DAG:     %[[IDX:.+]] = affine.apply #[[$XMAP]](%[[LANEID]])
//   CHECK-DAG:     %[[LHS_SLICE:.+]] = tensor.extract_slice %[[LHS]][0, 0, %[[IDX]], 0] [2, 8, 1, 16]
//   CHECK-DAG:     %[[RHS_SLICE:.+]] = tensor.extract_slice %[[RHS]][0, 0, %[[IDX]], 0] [8, 2, 1, 16]
//   CHECK-DAG:     %[[IDY:.+]] = affine.apply #[[$YMAP]](%[[LANEID]])
//   CHECK-DAG:     %[[ACC_SLICE:.+]] = tensor.extract_slice %[[ACC]][0, 0, 0, %[[IDY]], %[[IDX]]] [2, 2, 8, 1, 1]
//       CHECK:     %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS_SLICE]], %[[RHS_SLICE]], %[[ACC_SLICE]]
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<WMMA_F32_16x16x16_F16>
//  CHECK-SAME:       : tensor<2x8x1x16xf16>, tensor<8x2x1x16xf16> into tensor<2x2x8x1x1xf32>
//       CHECK:     tensor.parallel_insert_slice %[[MMA]] into %[[ACC]][0, 0, 0, %[[IDY]], %[[IDX]]] [2, 2, 8, 1, 1]
//       CHECK:   mapping = [#iree_gpu.lane_id<0>]
