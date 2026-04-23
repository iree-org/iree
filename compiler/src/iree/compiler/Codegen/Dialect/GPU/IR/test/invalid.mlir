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

// -----

// async_dma: wrong number of source indices.
#layout_err_idx = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 64],
  element_tile = [1, 1],
  subgroup_strides = [0, 0],
  thread_strides = [0, 1]
>

func.func @async_dma_wrong_index_count(%src: tensor<20x64xf16>,
                                        %dest: tensor<1x64xf16>,
                                        %i: index, %c0: index) {
  // expected-error @+1 {{expected 2 source indices (source rank), got 1}}
  %0 = iree_gpu.async_dma %src[%i] to %dest[%c0], #layout_err_idx
      : tensor<20x64xf16>, tensor<1x64xf16> -> tensor<1x64xf16>
  return
}

// -----

// async_dma: permutation_map required when ranks differ.
#layout_err_perm = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile = [1],
  outer_tile = [1],
  thread_tile = [64],
  element_tile = [1],
  subgroup_strides = [0],
  thread_strides = [1]
>

func.func @async_dma_missing_permutation_map(%src: memref<20x64xf16>,
                                              %dest: memref<64xf16>,
                                              %i: index, %j: index, %c0: index) {
  // expected-error @+1 {{permutation_map is required when source rank (2) differs from dest rank (1)}}
  iree_gpu.async_dma %src[%i, %j] to %dest[%c0], #layout_err_perm
      : memref<20x64xf16>, memref<64xf16>
  return
}

// -----

// async_dma: wrong in_bounds array size (checked against dest rank).
#layout_err_ib = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 64],
  element_tile = [1, 1],
  subgroup_strides = [0, 0],
  thread_strides = [0, 1]
>

func.func @async_dma_in_bounds_wrong_size(%src: tensor<20x64xf16>,
                                           %dest: tensor<1x64xf16>,
                                           %i: index, %j: index,
                                           %c0: index) {
  // expected-error @+1 {{in_bounds array size (1) must match dest rank (2)}}
  %0 = iree_gpu.async_dma %src[%i, %j] to %dest[%c0, %c0], #layout_err_ib
      in_bounds [true]
      : tensor<20x64xf16>, tensor<1x64xf16> -> tensor<1x64xf16>
  return
}

// -----

// async_dma: layout rank mismatch.
#layout_err_layout_rank = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile = [1],
  outer_tile = [1],
  thread_tile = [64],
  element_tile = [1],
  subgroup_strides = [0],
  thread_strides = [1]
>

func.func @async_dma_layout_rank_mismatch(%src: tensor<20x64xf16>,
                                            %dest: tensor<1x64xf16>,
                                            %i: index, %j: index,
                                            %c0: index) {
  // expected-error @+1 {{layout rank (1) must match dest rank (2)}}
  %0 = iree_gpu.async_dma %src[%i, %j] to %dest[%c0, %c0], #layout_err_layout_rank
      : tensor<20x64xf16>, tensor<1x64xf16> -> tensor<1x64xf16>
  return
}

// -----

// async_dma: permutation_map has wrong number of dims.
#layout_err_perm_dims = #iree_vector_ext.nested_layout<
  subgroup_tile = [1],
  batch_tile = [1],
  outer_tile = [1],
  thread_tile = [64],
  element_tile = [1],
  subgroup_strides = [0],
  thread_strides = [1]
>

func.func @async_dma_permutation_map_wrong_dims(%src: memref<20x64xf16>,
                                                  %dest: memref<64xf16>,
                                                  %i: index, %j: index, %c0: index) {
  // expected-error @+1 {{permutation_map num dims (1) must match source rank (2)}}
  iree_gpu.async_dma %src[%i, %j] to %dest[%c0], #layout_err_perm_dims
      permutation_map affine_map<(d0) -> (d0)>
      : memref<20x64xf16>, memref<64xf16>
  return
}

// -----

// async_dma: permutation_map has wrong number of results.
#layout_err_perm_results = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 64],
  element_tile = [1, 1],
  subgroup_strides = [0, 0],
  thread_strides = [0, 1]
>

func.func @async_dma_permutation_map_wrong_results(%src: memref<20x64xf16>,
                                                     %dest: memref<1x64xf16>,
                                                     %i: index, %j: index,
                                                     %c0: index) {
  // expected-error @+1 {{permutation_map num results (1) must match dest rank (2)}}
  iree_gpu.async_dma %src[%i, %j] to %dest[%c0, %c0], #layout_err_perm_results
      permutation_map affine_map<(d0, d1) -> (d0)>
      : memref<20x64xf16>, memref<1x64xf16>
  return
}

// -----

// async_dma: gather index size doesn't match layout dimension size.
#layout_err_gather_size = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile = [1, 1],
  outer_tile = [1, 1],
  thread_tile = [1, 64],
  element_tile = [1, 1],
  subgroup_strides = [0, 0],
  thread_strides = [0, 1]
>

func.func @async_dma_gather_size_mismatch(%src: tensor<1024x64xf16>,
                                           %dest: tensor<1x64xf16>,
                                           %indices: vector<4xindex>,
                                           %j: index, %c0: index) {
  // expected-error @+1 {{gather index size (4) for source dimension 0 must match layout size (1) in dest dimension 0}}
  %0 = iree_gpu.async_dma %src[%indices, %j] to %dest[%c0, %c0], #layout_err_gather_size
      : tensor<1024x64xf16> [vector<4xindex>, index],
        tensor<1x64xf16> -> tensor<1x64xf16>
  return
}
