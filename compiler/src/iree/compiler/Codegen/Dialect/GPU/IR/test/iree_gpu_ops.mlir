// RUN: iree-opt %s --split-input-file | FileCheck %s

func.func @barrier_region(%init: tensor<6x6xf32>) -> tensor<3x2xf32> {
  %0 = iree_gpu.barrier_region ins(%init : tensor<6x6xf32>) {
  ^bb0(%intermediate: tensor<6x6xf32>):
    %slice = tensor.extract_slice %intermediate[0, 0] [3, 2] [1, 1] : tensor<6x6xf32> to tensor<3x2xf32>
    iree_gpu.yield %slice : tensor<3x2xf32>
  } : tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// CHECK-LABEL: func @barrier_region
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9]+]]: tensor<6x6xf32>
//       CHECK:   iree_gpu.barrier_region ins(%[[INIT]] : tensor<6x6xf32>) {
//       CHECK:     ^bb0(%[[INTERMEDIATE:.+]]: tensor<6x6xf32>):
//       CHECK:       %[[SLICE:.+]] = tensor.extract_slice %[[INTERMEDIATE]][0, 0] [3, 2] [1, 1]
//       CHECK:       iree_gpu.yield %[[SLICE]] : tensor<3x2xf32>
//       CHECK:   } : tensor<3x2xf32>

// -----

func.func @multi_result_barrier_region(%init: tensor<12x12xf32>) -> (tensor<2x1x3x2xf32>, index) {
  %0:2 = iree_gpu.barrier_region ins(%init : tensor<12x12xf32>) {
  ^bb0(%intermediate: tensor<12x12xf32>):
    %expand = tensor.expand_shape %intermediate [[0, 1], [2, 3]] output_shape [4, 3, 3, 4] : tensor<12x12xf32> into tensor<4x3x3x4xf32>
    %slice = tensor.extract_slice %expand[0, 0, 0, 0] [2, 1, 3, 2] [1, 1, 1, 1] : tensor<4x3x3x4xf32> to tensor<2x1x3x2xf32>
    %c0 = arith.constant 0 : index
    iree_gpu.yield %slice, %c0 : tensor<2x1x3x2xf32>, index
  } : tensor<2x1x3x2xf32>, index
  return %0#0, %0#1 : tensor<2x1x3x2xf32>, index
}

// CHECK-LABEL: func @multi_result_barrier_region
//       CHECK:   %{{.*}}:2 = iree_gpu.barrier_region ins(%{{.*}} : tensor<12x12xf32>)
//       CHECK:   } : tensor<2x1x3x2xf32>, index

// -----

func.func @multi_input_barrier_region(%x: index, %y: index) -> index {
  %0 = iree_gpu.barrier_region ins(%x, %y : index, index) {
  ^bb0(%ix: index, %iy: index):
    %sum = arith.addi %ix, %iy : index
    iree_gpu.yield %sum : index
  } : index
  return %0 : index
}

// CHECK-LABEL: func @multi_input_barrier_region
//       CHECK:   %{{.*}} = iree_gpu.barrier_region ins(%{{.*}}, %{{.*}} : index, index)
//       CHECK:   } : index

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
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
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
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
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
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
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
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
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
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } : vector<4xf16>, vector<4xf16> into vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<() -> ()>

// CHECK-LABEL: func @single_multi_mma
//       CHECK:   iree_gpu.multi_mma %arg0, %arg1, %arg2
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]]
//  CHECK-SAME:       iterator_types = []
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
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
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
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
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>}
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
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
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
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
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
    kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
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
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
//  CHECK-SAME:     rhs_permutation = array<i64: 1, 0>}
//  CHECK-SAME:     : tensor<?x?x32x8xf16>, tensor<?x?x32x8xf16> into tensor<?x?x32x32xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @data_tiled_1x1x1_tensor_multi_mma(%lhs: tensor<?x?x4x16x1x1xf32>, %rhs: tensor<?x?x4x16x1x1xf32>, %acc: tensor<?x?x4x16x4x1xf32>) -> tensor<?x?x4x16x4x1xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32>
  } : tensor<?x?x4x16x1x1xf32>, tensor<?x?x4x16x1x1xf32> into tensor<?x?x4x16x4x1xf32>
  return %0 : tensor<?x?x4x16x4x1xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @data_tiled_1x1x1_tensor_multi_mma
//       CHECK:   iree_gpu.multi_mma %arg0, %arg1, %arg2
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>]
//  CHECK-SAME:       kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32>
//  CHECK-SAME:     : tensor<?x?x4x16x1x1xf32>, tensor<?x?x4x16x1x1xf32> into tensor<?x?x4x16x4x1xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @data_tiled_2x2x4_tensor_multi_mma(%lhs: tensor<?x?x2x4x16x1x4xf32>, %rhs: tensor<?x?x2x4x16x1x4xf32>, %acc: tensor<?x?x2x2x4x16x4x1xf32>) -> tensor<?x?x2x2x4x16x4x1xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, unroll_m = 2, unroll_n = 2, unroll_k = 4>
  } : tensor<?x?x2x4x16x1x4xf32>, tensor<?x?x2x4x16x1x4xf32> into tensor<?x?x2x2x4x16x4x1xf32>
  return %0 : tensor<?x?x2x2x4x16x4x1xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @data_tiled_2x2x4_tensor_multi_mma
//       CHECK:   iree_gpu.multi_mma %arg0, %arg1, %arg2
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>]
//  CHECK-SAME:       kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, unroll_m = 2, unroll_n = 2, unroll_k = 4>
//  CHECK-SAME:     : tensor<?x?x2x4x16x1x4xf32>, tensor<?x?x2x4x16x1x4xf32> into tensor<?x?x2x2x4x16x4x1xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @data_tiled_2x2x4_tensor_multi_mma(%lhs: tensor<?x?x2x4x16x1x4xf32>, %rhs: tensor<?x?x2x4x16x1x4xf32>, %acc: tensor<?x?x2x2x4x16x4x1xf32>) -> tensor<?x?x2x2x4x16x4x1xf32> {
  %0 = iree_gpu.multi_mma %lhs, %rhs, %acc {
    indexing_maps = #contraction_accesses,
    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, unroll_m_to_subgroups = 2, unroll_n_to_subgroups = 2, unroll_k = 4>
  } : tensor<?x?x2x4x16x1x4xf32>, tensor<?x?x2x4x16x1x4xf32> into tensor<?x?x2x2x4x16x4x1xf32>
  return %0 : tensor<?x?x2x2x4x16x4x1xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @data_tiled_2x2x4_tensor_multi_mma
//       CHECK:   iree_gpu.multi_mma %arg0, %arg1, %arg2
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>]
//  CHECK-SAME:       kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, unroll_m_to_subgroups = 2, unroll_n_to_subgroups = 2, unroll_k = 4>
//  CHECK-SAME:     : tensor<?x?x2x4x16x1x4xf32>, tensor<?x?x2x4x16x1x4xf32> into tensor<?x?x2x2x4x16x4x1xf32>


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

// -----

func.func @vector_barrier_multiple_inputs(%input: vector<8xf16>) -> (vector<8xf16>, vector<8xf16>) {
  %out:2 = iree_gpu.value_barrier %input, %input : vector<8xf16>, vector<8xf16>
  return %out#0, %out#1 : vector<8xf16>, vector<8xf16>
}

// CHECK-LABEL: func @vector_barrier_multiple_inputs
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: vector<8xf16>
//       CHECK:   iree_gpu.value_barrier %[[INPUT]], %[[INPUT]] : vector<8xf16>, vector<8xf16>

// -----

func.func @tensor_barrier_multiple_inputs(%input: tensor<?xf16>) -> (tensor<?xf16>, tensor<?xf16>) {
  %out:2 = iree_gpu.value_barrier %input, %input : tensor<?xf16>, tensor<?xf16>
  return %out#0, %out#1 : tensor<?xf16>, tensor<?xf16>
}

// CHECK-LABEL: func @tensor_barrier_multiple_inputs
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9]+]]: tensor<?xf16>
//       CHECK:   iree_gpu.value_barrier %[[INPUT]], %[[INPUT]] : tensor<?xf16>, tensor<?xf16>
