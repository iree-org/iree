// RUN: iree-opt %s --split-input-file | FileCheck %s

func.func @shuffle_tensor(%init: tensor<6x6xf32>, %source: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = iree_gpu.shuffle_tensor %source[0, 0] [2, 3] [1, 1] to %init {
  ^bb0(%intermediate: tensor<6x6xf32>):
    %slice = tensor.extract_slice %intermediate[0, 0] [3, 2] [1, 1] : tensor<6x6xf32> to tensor<3x2xf32>
    iree_gpu.yield %slice : tensor<3x2xf32>
  } : tensor<2x3xf32> -> tensor<6x6xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// CHECK-LABEL: func @shuffle_tensor
//       CHECK:   iree_gpu.shuffle_tensor %arg1[0, 0] [2, 3] [1, 1] to %arg0 {
//       CHECK:     ^bb0(%[[INTERMEDIATE:.+]]: tensor<6x6xf32>):
//       CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[INTERMEDIATE]][0, 0] [3, 2] [1, 1] : tensor<6x6xf32> to tensor<3x2xf32>
//       CHECK:       iree_gpu.yield %[[SLICE]] : tensor<3x2xf32>
//       CHECK:   } : tensor<2x3xf32> -> tensor<6x6xf32> -> tensor<3x2xf32>

// -----

func.func @rank_reducing_shuffle_tensor(%init: tensor<1x6x6xf32>, %source: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = iree_gpu.shuffle_tensor %source[0, 0, 0] [1, 2, 3] [1, 1, 1] to %init {
  ^bb0(%intermediate: tensor<1x6x6xf32>):
    %slice = tensor.extract_slice %intermediate[0, 0, 0] [1, 3, 2] [1, 1, 1] : tensor<1x6x6xf32> to tensor<3x2xf32>
    iree_gpu.yield %slice : tensor<3x2xf32>
  } : tensor<2x3xf32> -> tensor<1x6x6xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// CHECK-LABEL: func @rank_reducing_shuffle_tensor
//       CHECK:   iree_gpu.shuffle_tensor %arg1[0, 0, 0] [1, 2, 3] [1, 1, 1] to %arg0 {
//       CHECK:     ^bb0(%[[INTERMEDIATE:.+]]: tensor<1x6x6xf32>):
//       CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[INTERMEDIATE]][0, 0, 0] [1, 3, 2] [1, 1, 1] : tensor<1x6x6xf32> to tensor<3x2xf32>
//       CHECK:       iree_gpu.yield %[[SLICE]] : tensor<3x2xf32>
//       CHECK:   } : tensor<2x3xf32> -> tensor<1x6x6xf32> -> tensor<3x2xf32>

// -----

func.func @reshape_shuffle_tensor(%init: tensor<12x12xf32>, %source: tensor<2x3xf32>) -> tensor<2x1x3x2xf32> {
  %0 = iree_gpu.shuffle_tensor %source[0, 0] [2, 3] [1, 1] to %init {
  ^bb0(%intermediate: tensor<12x12xf32>):
    %expand = tensor.expand_shape %intermediate [[0, 1], [2, 3]] output_shape [4, 3, 3, 4] : tensor<12x12xf32> into tensor<4x3x3x4xf32>
    %slice = tensor.extract_slice %expand[0, 0, 0, 0] [2, 1, 3, 2] [1, 1, 1, 1] : tensor<4x3x3x4xf32> to tensor<2x1x3x2xf32>
    iree_gpu.yield %slice : tensor<2x1x3x2xf32>
  } : tensor<2x3xf32> -> tensor<12x12xf32> -> tensor<2x1x3x2xf32>
  return %0 : tensor<2x1x3x2xf32>
}

// CHECK-LABEL: func @reshape_shuffle_tensor
//       CHECK:   iree_gpu.shuffle_tensor
//       CHECK:       tensor.expand_shape
//       CHECK:       tensor.extract_slice
//       CHECK:   } : tensor<2x3xf32> -> tensor<12x12xf32> -> tensor<2x1x3x2xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @vector_multi_mma(%lhs: vector<2x3x4xf16>, %rhs: vector<3x5x4xf16>, %acc: vector<2x5x4xf32>) -> vector<2x5x4xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
  } : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>
  return %0 : vector<2x5x4xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @vector_multi_mma
//       CHECK:   iree_gpu.multi_mma %arg0, %arg1, %arg2
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
//  CHECK-SAME:     : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @tensor_multi_mma(%lhs: tensor<?x?x4xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
  } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @tensor_multi_mma
//       CHECK:   iree_gpu.multi_mma %arg0, %arg1, %arg2
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
//  CHECK-SAME:     : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @single_multi_mma(%lhs: vector<4xf16>, %rhs: vector<4xf16>, %acc: vector<4xf32>) -> vector<4xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
  } : vector<4xf16>, vector<4xf16> into vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<() -> ()>

// CHECK-LABEL: func @single_multi_mma
//       CHECK:   iree_gpu.multi_mma %arg0, %arg1, %arg2
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]]
//  CHECK-SAME:       iterator_types = []
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
//  CHECK-SAME:     : vector<4xf16>, vector<4xf16> into vector<4xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @tensor_subgroup_multi_mma(%lhs: tensor<?x?x16x16xf16>, %rhs: tensor<?x?x16x16xf16>, %acc: tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>
  } : tensor<?x?x16x16xf16>, tensor<?x?x16x16xf16> into tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @tensor_subgroup_multi_mma
//       CHECK:   iree_gpu.multi_mma %arg0, %arg1, %arg2
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]],
//  CHECK-SAME:     iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>}
//  CHECK-SAME:     : tensor<?x?x16x16xf16>, tensor<?x?x16x16xf16> into tensor<?x?x16x16xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @tensor_subgroup_matmul_transpose_b_multi_mma(%lhs: tensor<?x?x16x16xf16>, %rhs: tensor<?x?x16x16xf16>, %acc: tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
    rhs_permutation = array<i64: 1, 0>
  } : tensor<?x?x16x16xf16>, tensor<?x?x16x16xf16> into tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @tensor_subgroup_matmul_transpose_b_multi_mma
//       CHECK:   iree_gpu.multi_mma %arg0, %arg1, %arg2
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]],
//  CHECK-SAME:     iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F16_16x16x16_F32>,
//  CHECK-SAME:     rhs_permutation = array<i64: 1, 0>}
//  CHECK-SAME:     : tensor<?x?x16x16xf16>, tensor<?x?x16x16xf16> into tensor<?x?x16x16xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @tensor_subgroup_matmul_transpose_b_32x32x8_multi_mma(
  %lhs: tensor<?x?x32x8xf16>,
  %rhs: tensor<?x?x32x8xf16>,
  %acc: tensor<?x?x32x32xf32>) -> tensor<?x?x32x32xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>,
    rhs_permutation = array<i64: 1, 0>
  } : tensor<?x?x32x8xf16>, tensor<?x?x32x8xf16> into tensor<?x?x32x32xf32>
  return %0 : tensor<?x?x32x32xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @tensor_subgroup_matmul_transpose_b_32x32x8_multi_mma
//       CHECK:   iree_gpu.multi_mma %arg0, %arg1, %arg2
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]],
//  CHECK-SAME:     iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>,
//  CHECK-SAME:     rhs_permutation = array<i64: 1, 0>}
//  CHECK-SAME:     : tensor<?x?x32x8xf16>, tensor<?x?x32x8xf16> into tensor<?x?x32x32xf32>

// -----

func.func @tensor_barrier(%input: tensor<?xf16>) -> tensor<?xf16> {
  %out = iree_gpu.value_barrier %input : tensor<?xf16>
  return %out : tensor<?xf16>
}

// CHECK-LABEL: func @tensor_barrier
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<?xf16>
//       CHECK:   iree_gpu.value_barrier %[[INPUT]] : tensor<?xf16>

// -----

func.func @vector_barrier(%input: vector<8xf16>) -> vector<8xf16> {
  %out = iree_gpu.value_barrier %input : vector<8xf16>
  return %out : vector<8xf16>
}

// CHECK-LABEL: func @vector_barrier
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: vector<8xf16>
//       CHECK:   iree_gpu.value_barrier %[[INPUT]] : vector<8xf16>
