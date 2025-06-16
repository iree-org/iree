// RUN: iree-opt %s -split-input-file | FileCheck %s

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @vector_multi_mma(%lhs: vector<2x3x4xf16>, %rhs: vector<3x5x4xf16>, %acc: vector<2x5x4xf32>) -> vector<2x5x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>
  return %0 : vector<2x5x4xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @vector_multi_mma
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
//  CHECK-SAME:     : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @tensor_multi_mma(%lhs: tensor<?x?x4xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @tensor_multi_mma
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
//  CHECK-SAME:       kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
//  CHECK-SAME:     : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @single_multi_mma(%lhs: vector<4xf16>, %rhs: vector<4xf16>, %acc: vector<4xf32>) -> vector<4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } : vector<4xf16>, vector<4xf16> into vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<() -> ()>

// CHECK-LABEL: func @single_multi_mma
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
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
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } : tensor<?x?x16x16xf16>, tensor<?x?x16x16xf16> into tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @tensor_subgroup_multi_mma
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]],
//  CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>}
//  CHECK-SAME:     : tensor<?x?x16x16xf16>, tensor<?x?x16x16xf16> into tensor<?x?x16x16xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @tensor_subgroup_matmul_transpose_b_multi_mma(%lhs: tensor<?x?x16x16xf16>, %rhs: tensor<?x?x16x16xf16>, %acc: tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]
  } : tensor<?x?x16x16xf16>, tensor<?x?x16x16xf16> into tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @tensor_subgroup_matmul_transpose_b_multi_mma
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]],
//  CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
//  CHECK-SAME:     permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]}
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
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
    permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]
  } : tensor<?x?x32x8xf16>, tensor<?x?x32x8xf16> into tensor<?x?x32x32xf32>
  return %0 : tensor<?x?x32x32xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @tensor_subgroup_matmul_transpose_b_32x32x8_multi_mma
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]],
//  CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
//  CHECK-SAME:     kind = #iree_gpu.mma_layout<MFMA_F32_32x32x8_F16>,
//  CHECK-SAME:     permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]}
//  CHECK-SAME:     : tensor<?x?x32x8xf16>, tensor<?x?x32x8xf16> into tensor<?x?x32x32xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @data_tiled_1x1x1_tensor_multi_mma(%lhs: tensor<?x?x4x16x1x1xf32>, %rhs: tensor<?x?x4x16x1x1xf32>, %acc: tensor<?x?x4x16x4x1xf32>) -> tensor<?x?x4x16x4x1xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32>
  } : tensor<?x?x4x16x1x1xf32>, tensor<?x?x4x16x1x1xf32> into tensor<?x?x4x16x4x1xf32>
  return %0 : tensor<?x?x4x16x4x1xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @data_tiled_1x1x1_tensor_multi_mma
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
//  CHECK-SAME:       kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32>
//  CHECK-SAME:     : tensor<?x?x4x16x1x1xf32>, tensor<?x?x4x16x1x1xf32> into tensor<?x?x4x16x4x1xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @data_tiled_2x2x4_tensor_multi_mma(%lhs: tensor<?x?x2x4x16x1x4xf32>, %rhs: tensor<?x?x2x4x16x1x4xf32>, %acc: tensor<?x?x2x2x4x16x4x1xf32>) -> tensor<?x?x2x2x4x16x4x1xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, intrinsics_m = 2, intrinsics_n = 2, intrinsics_k = 4>
  } : tensor<?x?x2x4x16x1x4xf32>, tensor<?x?x2x4x16x1x4xf32> into tensor<?x?x2x2x4x16x4x1xf32>
  return %0 : tensor<?x?x2x2x4x16x4x1xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @data_tiled_2x2x4_tensor_multi_mma
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
//  CHECK-SAME:       kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, intrinsics_m = 2, intrinsics_n = 2, intrinsics_k = 4>
//  CHECK-SAME:     : tensor<?x?x2x4x16x1x4xf32>, tensor<?x?x2x4x16x1x4xf32> into tensor<?x?x2x2x4x16x4x1xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @data_tiled_2x2x4_tensor_multi_mma(%lhs: tensor<?x?x2x4x16x1x4xf32>, %rhs: tensor<?x?x2x4x16x1x4xf32>, %acc: tensor<?x?x2x2x4x16x4x1xf32>) -> tensor<?x?x2x2x4x16x4x1xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, subgroups_m = 2, subgroups_n = 2, intrinsics_k = 4>
  } : tensor<?x?x2x4x16x1x4xf32>, tensor<?x?x2x4x16x1x4xf32> into tensor<?x?x2x2x4x16x4x1xf32>
  return %0 : tensor<?x?x2x2x4x16x4x1xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @data_tiled_2x2x4_tensor_multi_mma
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%arg2)
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]]]
//  CHECK-SAME:       iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
//  CHECK-SAME:       kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, subgroups_m = 2, subgroups_n = 2, intrinsics_k = 4>
//  CHECK-SAME:     : tensor<?x?x2x4x16x1x4xf32>, tensor<?x?x2x4x16x1x4xf32> into tensor<?x?x2x2x4x16x4x1xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k, b) -> (i, k, b)>,
 affine_map<(i, j, k, b) -> (i, k)>,
 affine_map<(i, j, k, b) -> (k, b, j)>,
 affine_map<(i, j, k, b) -> (k, j)>,
 affine_map<(i, j, k, b) -> (i, j)>
]
func.func @vector_scaled_multi_mma(%lhs: vector<2x3x1x32xf4E2M1FN>, %lhsScale: vector<2x3x1xf8E8M0FNU>,
    %rhs: vector<3x1x5x32xf8E4M3FN>, %rhsScale: vector<3x5x1xf8E8M0FNU>,
    %acc: vector<2x5x4xf32>) -> vector<2x5x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %lhsScale, %rhs, %rhsScale) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.scaled_mma_layout<
      intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN,
      rhs_elem_type = f8E4M3FN,
      acc_elem_type = f32>
  } : vector<2x3x1x32xf4E2M1FN>, vector<2x3x1xf8E8M0FNU>,
    vector<3x1x5x32xf8E4M3FN>, vector<3x5x1xf8E8M0FNU>
    into vector<2x5x4xf32>
  return %0 : vector<2x5x4xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1)>
// CHECK: #[[$MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// CHECK-LABEL: func @vector_scaled_multi_mma
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1, %arg2, %arg3) outs(%arg4)
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]], #[[$MAP3]], #[[$MAP4]]]
//  CHECK-SAME:       iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>]
//  CHECK-SAME:       kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>
//  CHECK-SAME:     : vector<2x3x1x32xf4E2M1FN>, vector<2x3x1xf8E8M0FNU>, vector<3x1x5x32xf8E4M3FN>, vector<3x5x1xf8E8M0FNU> into vector<2x5x4xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k, b) -> (i, k, b)>,
 affine_map<(i, j, k, b) -> (i, k)>,
 affine_map<(i, j, k, b) -> (k, b, j)>,
 affine_map<(i, j, k, b) -> (k, j)>,
 affine_map<(i, j, k, b) -> (i, j)>
]

func.func @tensor_scaled_multi_mma(%lhs: tensor<?x?x1x32xf4E2M1FN>, %lhsScale: tensor<?x?x1xf8E8M0FNU>,
    %rhs: tensor<?x1x?x32xf8E4M3FN>, %rhsScale: tensor<?x?x1xf8E8M0FNU>,
    %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %lhsScale, %rhs, %rhsScale) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.scaled_mma_layout<
      intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN,
      rhs_elem_type = f8E4M3FN,
      acc_elem_type = f32>
  } : tensor<?x?x1x32xf4E2M1FN>, tensor<?x?x1xf8E8M0FNU>,
    tensor<?x1x?x32xf8E4M3FN>, tensor<?x?x1xf8E8M0FNU>
    into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1)>
// CHECK: #[[$MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// CHECK-LABEL: func @tensor_scaled_multi_mma
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1, %arg2, %arg3) outs(%arg4)
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]], #[[$MAP3]], #[[$MAP4]]]
//  CHECK-SAME:       iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>]
//  CHECK-SAME:       kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>
//  CHECK-SAME:     : tensor<?x?x1x32xf4E2M1FN>, tensor<?x?x1xf8E8M0FNU>, tensor<?x1x?x32xf8E4M3FN>, tensor<?x?x1xf8E8M0FNU> into tensor<?x?x4xf32>

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @single_scaled_multi_mma(%lhs: vector<32xf4E2M1FN>, %lhsScale: vector<1xf8E8M0FNU>,
    %rhs: vector<32xf8E4M3FN>, %rhsScale: vector<1xf8E8M0FNU>,
    %acc: vector<4xf32>) -> vector<4xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %lhsScale, %rhs, %rhsScale) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.scaled_mma_layout<
      intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN,
      rhs_elem_type = f8E4M3FN,
      acc_elem_type = f32>
  } : vector<32xf4E2M1FN>, vector<1xf8E8M0FNU>,
    vector<32xf8E4M3FN>, vector<1xf8E8M0FNU>
    into vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<() -> ()>

// CHECK-LABEL: func @single_scaled_multi_mma
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1, %arg2, %arg3) outs(%arg4)
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]], #[[$MAP]], #[[$MAP]]]
//  CHECK-SAME:       iterator_types = []
//  CHECK-SAME:       kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>
//  CHECK-SAME:     : vector<32xf4E2M1FN>, vector<1xf8E8M0FNU>, vector<32xf8E4M3FN>, vector<1xf8E8M0FNU> into vector<4xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k, b) -> (i, k, b)>,
 affine_map<(i, j, k, b) -> (i, k)>,
 affine_map<(i, j, k, b) -> (k, b, j)>,
 affine_map<(i, j, k, b) -> (k, j)>,
 affine_map<(i, j, k, b) -> (i, j)>
]

func.func @tensor_subgroup_scaled_multi_mma(%lhs: tensor<?x?x2x16x4x32xf4E2M1FN>, %lhsScale: tensor<?x?x16x4xf8E8M0FNU>,
    %rhs: tensor<?x2x?x4x32x16xf8E4M3FN>, %rhsScale: tensor<?x?x4x16xf8E8M0FNU>,
    %acc: tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32> {
  %0 = iree_codegen.inner_tiled ins(%lhs, %lhsScale, %rhs, %rhsScale) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.scaled_mma_layout<
      intrinsic = MFMA_SCALE_F32_16x16x128_B32,
      lhs_elem_type = f4E2M1FN,
      rhs_elem_type = f8E4M3FN,
      acc_elem_type = f32>
  } : tensor<?x?x2x16x4x32xf4E2M1FN>, tensor<?x?x16x4xf8E8M0FNU>,
    tensor<?x2x?x4x32x16xf8E4M3FN>, tensor<?x?x4x16xf8E8M0FNU>
    into tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1)>
// CHECK: #[[$MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// CHECK-LABEL: func @tensor_subgroup_scaled_multi_mma
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1, %arg2, %arg3) outs(%arg4)
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]], #[[$MAP3]], #[[$MAP4]]],
//  CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
//  CHECK-SAME:     kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>
//  CHECK-SAME:     : tensor<?x?x2x16x4x32xf4E2M1FN>, tensor<?x?x16x4xf8E8M0FNU>, tensor<?x2x?x4x32x16xf8E4M3FN>, tensor<?x?x4x16xf8E8M0FNU> into tensor<?x?x16x16xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k, b) -> (i, k, b)>,
 affine_map<(i, j, k, b) -> (i, k)>,
 affine_map<(i, j, k, b) -> (k, b, j)>,
 affine_map<(i, j, k, b) -> (k, j)>,
 affine_map<(i, j, k, b) -> (i, j)>
]

func.func @tensor_subgroup_scaled_matmul_transpose_b_multi_mma(%lhs: tensor<?x?x1x16x4x32xf4E2M1FN>, %lhsScale: tensor<?x?x16x4xf8E8M0FNU>,
    %rhs: tensor<?x1x?x16x4x32xf8E4M3FN>, %rhsScale: tensor<?x?x16x4xf8E8M0FNU>,
    %acc: tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32> {
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
  } : tensor<?x?x1x16x4x32xf4E2M1FN>, tensor<?x?x16x4xf8E8M0FNU>,
    tensor<?x1x?x16x4x32xf8E4M3FN>, tensor<?x?x16x4xf8E8M0FNU>
    into tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1)>
// CHECK: #[[$MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// CHECK-LABEL: func @tensor_subgroup_scaled_matmul_transpose_b_multi_mma
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1, %arg2, %arg3) outs(%arg4)
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]], #[[$MAP3]], #[[$MAP4]]],
//  CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
//  CHECK-SAME:     kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>,
//  CHECK-SAME:     permutations = [array<i64: 0, 1, 2>, array<i64: 0, 1>, array<i64: 2, 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]
//  CHECK-SAME:     : tensor<?x?x1x16x4x32xf4E2M1FN>, tensor<?x?x16x4xf8E8M0FNU>, tensor<?x1x?x16x4x32xf8E4M3FN>, tensor<?x?x16x4xf8E8M0FNU> into tensor<?x?x16x16xf32>

// -----

#contraction_accesses = [
 affine_map<(i, j, k, b) -> (i, k, b)>,
 affine_map<(i, j, k, b) -> (i, k)>,
 affine_map<(i, j, k, b) -> (k, b, j)>,
 affine_map<(i, j, k, b) -> (k, j)>,
 affine_map<(i, j, k, b) -> (i, j)>
]

func.func @tensor_subgroup_scaled_matmul_transpose_b_32x32x64_multi_mma(%lhs: tensor<?x?x1x32x2x32xf4E2M1FN>, %lhsScale: tensor<?x?x32x2xf8E8M0FNU>,
    %rhs: tensor<?x1x?x32x2x32xf8E4M3FN>, %rhsScale: tensor<?x?x32x2xf8E8M0FNU>,
    %acc: tensor<?x?x32x32xf32>) -> tensor<?x?x32x32xf32> {
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
      array<i64: 0, 1>]
  } : tensor<?x?x1x32x2x32xf4E2M1FN>, tensor<?x?x32x2xf8E8M0FNU>,
    tensor<?x1x?x32x2x32xf8E4M3FN>, tensor<?x?x32x2xf8E8M0FNU>
    into tensor<?x?x32x32xf32>
  return %0 : tensor<?x?x32x32xf32>
}

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>
// CHECK: #[[$MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1)>
// CHECK: #[[$MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// CHECK-LABEL: func @tensor_subgroup_scaled_matmul_transpose_b_32x32x64_multi_mma
//       CHECK:   iree_codegen.inner_tiled ins(%arg0, %arg1, %arg2, %arg3) outs(%arg4)
//  CHECK-SAME:     indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP2]], #[[$MAP3]], #[[$MAP4]]],
//  CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>],
//  CHECK-SAME:     kind = #iree_gpu.scaled_mma_layout<intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32>,
//  CHECK-SAME:     permutations = [array<i64: 0, 1, 2>, array<i64: 0, 1>, array<i64: 2, 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]
//  CHECK-SAME:     : tensor<?x?x1x32x2x32xf4E2M1FN>, tensor<?x?x32x2xf8E8M0FNU>, tensor<?x1x?x32x2x32xf8E4M3FN>, tensor<?x?x32x2xf8E8M0FNU> into tensor<?x?x32x32xf32>
