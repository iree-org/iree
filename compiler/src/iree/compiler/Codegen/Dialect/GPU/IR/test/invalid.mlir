// RUN: iree-opt --split-input-file --verify-diagnostics %s

func.func @mma_inner_tiled_invalid_num_inputs(%lhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  // expected-error @+1 {{number of inputs (1) doesn't match expected number from kind (2)}}
  %0 = iree_codegen.inner_tiled ins(%lhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_num_outputs(%lhs: tensor<?x?x4xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> (tensor<?x?x4xf32>, tensor<?x?x4xf32>) {
  // expected-error @+1 {{number of outputs (2) doesn't match expected number from kind (1)}}
  %0:2 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc, %acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>, tensor<?x?x4xf32>
  return %0#0, %0#1 : tensor<?x?x4xf32>, tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_num_indexing_maps(%lhs: tensor<?x?x4xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  // expected-error @+1 {{expected an indexing map for each operand}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_indexing_map_num_dims(%lhs: tensor<?x?x4xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  // expected-error @+1 {{expected indexing map 0 to have 3 input dims}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k, x) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_indexing_map_num_results(%lhs: tensor<?x?x4xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  // expected-error @+1 {{expected indexing map 0 to have fewer than 3 results}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, j, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_indexing_map_non_permutation(%lhs: tensor<?x?x4xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  // expected-error @+1 {{expected indexing map 0 to be a projected permutation}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, j + k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_outer_shape(%lhs: tensor<2x2x4xf16>, %rhs: tensor<2x3x4xf16>, %acc: tensor<2x2x4xf32>) -> tensor<2x2x4xf32> {
  // expected-error @+1 {{shape does not match iteration bounds}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<2x2x4xf16>, tensor<2x3x4xf16> into tensor<2x2x4xf32>
  return %0 : tensor<2x2x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_dynamic_inner_dim(%lhs: tensor<?x?x?xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  // expected-error @+1 {{Unexpected dynamic inner dim for operand 0 of type 'tensor<?x?x?xf16>'}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<?x?x?xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_element_type(%lhs: tensor<?x?x4xf32>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  // expected-error @+1 {{op operand element type f32 does not match expected MMA tile element type f16}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<?x?x4xf32>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_inner_types_distributed_opaque(%lhs: tensor<?x?x3xf16>, %rhs: tensor<?x?x4xf16>, %acc: tensor<?x?x4xf32>) -> tensor<?x?x4xf32> {
  // expected-error @+1 {{op operand type tensor<?x?x3xf16>, implying tile type vector<3xf16>, is incompatible with permuted InnerTiledDescAttr tile type vector<4xf16> under semantics #iree_gpu.mma_semantics<distributed = true, opaque = true>}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>
  } : tensor<?x?x3xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %0 : tensor<?x?x4xf32>
}

// -----

func.func @mma_inner_tiled_invalid_inner_types_undistributed_nonopaque(%lhs: tensor<?x?x4x16x4xf16>, %rhs: tensor<?x?x4x16x4xf16>, %acc: tensor<?x?x16x16xf32>) -> tensor<?x?x16x16xf32> {
  // expected-error @+1 {{op operand type tensor<?x?x16x16xf32>, implying tile type vector<16x16xf32>, is incompatible with permuted InnerTiledDescAttr tile type vector<4x16x4xf32> under semantics #iree_gpu.mma_semantics<distributed = false, opaque = false>}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = false>
  } : tensor<?x?x4x16x4xf16>, tensor<?x?x4x16x4xf16> into tensor<?x?x16x16xf32>
  return %0 : tensor<?x?x16x16xf32>
}

// -----

func.func @mma_inner_tiled_invalid_inner_types_distributed_nonopaque(%lhs: tensor<?x?x1x1x4xf16>, %rhs: tensor<?x?x1x1x4xf16>, %acc: tensor<?x?x1x2x2xf32>) -> tensor<?x?x1x2x2xf32> {
  // expected-error @+1 {{op operand type tensor<?x?x1x2x2xf32>, implying tile type vector<1x2x2xf32>, is incompatible with permuted InnerTiledDescAttr tile type vector<1x1x4xf32> under semantics #iree_gpu.mma_semantics<distributed = true, opaque = false>}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [affine_map<(i, j, k) -> (i, k)>, affine_map<(i, j, k) -> (k, j)>, affine_map<(i, j, k) -> (i, j)>],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : tensor<?x?x1x1x4xf16>, tensor<?x?x1x1x4xf16> into tensor<?x?x1x2x2xf32>
  return %0 : tensor<?x?x1x2x2xf32>
}

// -----

func.func @vector_multi_mma_with_wrong_number_of_permutations(%lhs: vector<2x3x4xf16>, %rhs: vector<3x5x4xf16>, %acc: vector<2x5x4xf32>) -> vector<2x5x4xf32> {
  // expected-error @+1 {{op mismatch between the number of permutations (2) and the number of operands (3)}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [
      affine_map<(i, j, k) -> (i, k)>,
      affine_map<(i, j, k) -> (k, j)>,
      affine_map<(i, j, k) -> (i, j)>
    ],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>,
    permutations = [array<i64: 0, 1>, array<i64: 1, 0>]
  } : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>
  return %0 : vector<2x5x4xf32>
}

// -----

func.func @vector_multi_mma_with_permutation_of_wrong_size(%lhs: vector<2x3x4xf16>, %rhs: vector<3x5x4xf16>, %acc: vector<2x5x4xf32>) -> vector<2x5x4xf32> {
  // expected-error @+1 {{op permutation #0 length 2 does not match the inner rank 1 of the corresponding operand of type vector<2x3x4xf16>}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = [
      affine_map<(i, j, k) -> (i, k)>,
      affine_map<(i, j, k) -> (k, j)>,
      affine_map<(i, j, k) -> (i, j)>
    ],
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>,
    permutations = [array<i64: 0, 1>, array<i64: 1, 0>, array<i64: 0, 1>]
  } : vector<2x3x4xf16>, vector<3x5x4xf16> into vector<2x5x4xf32>
  return %0 : vector<2x5x4xf32>
}
