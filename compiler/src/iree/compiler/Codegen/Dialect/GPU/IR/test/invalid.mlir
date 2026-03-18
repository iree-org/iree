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

func.func @arg_compare_dimension_out_of_range(%input: vector<4x8xf32>,
                                               %init_val: vector<4xf32>,
                                               %init_idx: vector<4xi32>) {
  // expected-error @+1 {{dimension 2 is out of range [0, 2)}}
  %val, %idx = iree_gpu.arg_compare {2}(%input : vector<4x8xf32>)
                 inits(%init_val, %init_idx : vector<4xf32>, vector<4xi32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf ogt, %lhs, %rhs : f32
    iree_gpu.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi32>
  return
}

// -----

func.func @arg_compare_negative_dimension(%input: vector<4x8xf32>,
                                           %init_val: vector<4xf32>,
                                           %init_idx: vector<4xi32>) {
  // expected-error @+1 {{dimension -1 is out of range [0, 2)}}
  %val, %idx = iree_gpu.arg_compare {-1}(%input : vector<4x8xf32>)
                 inits(%init_val, %init_idx : vector<4xf32>, vector<4xi32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf ogt, %lhs, %rhs : f32
    iree_gpu.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi32>
  return
}

// -----

func.func @arg_compare_wrong_init_value_rank(%input: vector<4x8xf32>,
                                              %init_val: vector<4x8xf32>,
                                              %init_idx: vector<4xi32>) {
  // expected-error @+1 {{init value rank (2) must be input rank - 1 (1)}}
  %val, %idx = iree_gpu.arg_compare {1}(%input : vector<4x8xf32>)
                 inits(%init_val, %init_idx : vector<4x8xf32>, vector<4xi32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf ogt, %lhs, %rhs : f32
    iree_gpu.yield %cmp : i1
  } -> vector<4x8xf32>, vector<4xi32>
  return
}

// -----

func.func @arg_compare_wrong_init_index_rank(%input: vector<4x8xf32>,
                                              %init_val: vector<4xf32>,
                                              %init_idx: vector<4x8xi32>) {
  // expected-error @+1 {{init index rank (2) must be input rank - 1 (1)}}
  %val, %idx = iree_gpu.arg_compare {1}(%input : vector<4x8xf32>)
                 inits(%init_val, %init_idx : vector<4xf32>, vector<4x8xi32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf ogt, %lhs, %rhs : f32
    iree_gpu.yield %cmp : i1
  } -> vector<4xf32>, vector<4x8xi32>
  return
}

// -----

func.func @arg_compare_wrong_init_value_shape(%input: vector<4x8xf32>,
                                               %init_val: vector<8xf32>,
                                               %init_idx: vector<4xi32>) {
  // expected-error @+1 {{init value shape must match input shape with reduction dimension removed. Expected: [4], but got: [8]}}
  %val, %idx = iree_gpu.arg_compare {1}(%input : vector<4x8xf32>)
                 inits(%init_val, %init_idx : vector<8xf32>, vector<4xi32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf ogt, %lhs, %rhs : f32
    iree_gpu.yield %cmp : i1
  } -> vector<8xf32>, vector<4xi32>
  return
}

// -----

func.func @arg_compare_wrong_init_index_shape(%input: vector<4x8xf32>,
                                               %init_val: vector<4xf32>,
                                               %init_idx: vector<8xi32>) {
  // expected-error @+1 {{init index shape must match input shape with reduction dimension removed. Expected: [4], but got: [8]}}
  %val, %idx = iree_gpu.arg_compare {1}(%input : vector<4x8xf32>)
                 inits(%init_val, %init_idx : vector<4xf32>, vector<8xi32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf ogt, %lhs, %rhs : f32
    iree_gpu.yield %cmp : i1
  } -> vector<4xf32>, vector<8xi32>
  return
}

// -----

func.func @arg_compare_wrong_index_element_type(%input: vector<4x8xf32>,
                                                 %init_val: vector<4xf32>,
                                                 %init_idx: vector<4xf32>) {
  // expected-error @+1 {{init index must have integer or index element type, but got 'f32'}}
  %val, %idx = iree_gpu.arg_compare {1}(%input : vector<4x8xf32>)
                 inits(%init_val, %init_idx : vector<4xf32>, vector<4xf32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf ogt, %lhs, %rhs : f32
    iree_gpu.yield %cmp : i1
  } -> vector<4xf32>, vector<4xf32>
  return
}

// -----

func.func @arg_compare_explicit_index_shape_mismatch(%val: vector<2x4xf32>,
                                                      %idx: vector<2x8xi32>,
                                                      %init_val: vector<2xf32>,
                                                      %init_idx: vector<2xi32>) {
  // expected-error @+1 {{explicit-index mode: value and index inputs must have the same shape. Value shape: [2, 4], index shape: [2, 8]}}
  %result_val, %result_idx = iree_gpu.arg_compare {1}(%val, %idx : vector<2x4xf32>, vector<2x8xi32>)
                               inits(%init_val, %init_idx : vector<2xf32>, vector<2xi32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf ogt, %lhs, %rhs : f32
    iree_gpu.yield %cmp : i1
  } -> vector<2xf32>, vector<2xi32>
  return
}

// -----

func.func @arg_compare_explicit_index_wrong_element_type(%val: vector<2x4xf32>,
                                                          %idx: vector<2x4xf32>,
                                                          %init_val: vector<2xf32>,
                                                          %init_idx: vector<2xi32>) {
  // expected-error @+1 {{explicit-index mode: index input must have integer or index element type, but got 'f32'}}
  %result_val, %result_idx = iree_gpu.arg_compare {1}(%val, %idx : vector<2x4xf32>, vector<2x4xf32>)
                               inits(%init_val, %init_idx : vector<2xf32>, vector<2xi32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf ogt, %lhs, %rhs : f32
    iree_gpu.yield %cmp : i1
  } -> vector<2xf32>, vector<2xi32>
  return
}

// -----

func.func @arg_compare_explicit_index_type_mismatch(%val: vector<2x4xf32>,
                                                     %idx: vector<2x4xi64>,
                                                     %init_val: vector<2xf32>,
                                                     %init_idx: vector<2xi32>) {
  // expected-error @+1 {{explicit-index mode: input and init index element types must match. Input index type: 'i64', init index type: 'i32'}}
  %result_val, %result_idx = iree_gpu.arg_compare {1}(%val, %idx : vector<2x4xf32>, vector<2x4xi64>)
                               inits(%init_val, %init_idx : vector<2xf32>, vector<2xi32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf ogt, %lhs, %rhs : f32
    iree_gpu.yield %cmp : i1
  } -> vector<2xf32>, vector<2xi32>
  return
}

// -----

func.func @arg_compare_index_base_with_explicit_index(%val: vector<2x4xf32>,
                                                       %idx: vector<2x4xi32>,
                                                       %init_val: vector<2xf32>,
                                                       %init_idx: vector<2xi32>,
                                                       %base: i32) {
  // expected-error @+1 {{index_base must not be used with explicit indices}}
  %result_val, %result_idx = iree_gpu.arg_compare {1}(%val, %idx : vector<2x4xf32>, vector<2x4xi32>)
                               inits(%init_val, %init_idx : vector<2xf32>, vector<2xi32>)
                               index_base %base : i32 {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf ogt, %lhs, %rhs : f32
    iree_gpu.yield %cmp : i1
  } -> vector<2xf32>, vector<2xi32>
  return
}

// -----

func.func @arg_compare_wrong_num_comparator_args(%input: vector<4x8xf32>,
                                                  %init_val: vector<4xf32>,
                                                  %init_idx: vector<4xi32>) {
  // expected-error @+1 {{comparator region must have exactly 2 arguments}}
  %val, %idx = iree_gpu.arg_compare {1}(%input : vector<4x8xf32>)
                 inits(%init_val, %init_idx : vector<4xf32>, vector<4xi32>) {
  ^bb0(%lhs: f32):
    %c_true = arith.constant true
    iree_gpu.yield %c_true : i1
  } -> vector<4xf32>, vector<4xi32>
  return
}

// -----

func.func @arg_compare_wrong_comparator_arg_type(%input: vector<4x8xf32>,
                                                  %init_val: vector<4xf32>,
                                                  %init_idx: vector<4xi32>) {
  // expected-error @+1 {{comparator arguments must match input value element type. Expected: 'f32', but got: 'i32' and 'i32'}}
  %val, %idx = iree_gpu.arg_compare {1}(%input : vector<4x8xf32>)
                 inits(%init_val, %init_idx : vector<4xf32>, vector<4xi32>) {
  ^bb0(%lhs: i32, %rhs: i32):
    %cmp = arith.cmpi slt, %lhs, %rhs : i32
    iree_gpu.yield %cmp : i1
  } -> vector<4xf32>, vector<4xi32>
  return
}

// -----

func.func @arg_compare_wrong_yield_type(%input: vector<4x8xf32>,
                                        %init_val: vector<4xf32>,
                                        %init_idx: vector<4xi32>) {
  // expected-error @+1 {{comparator region must yield i1, but got 'f32'}}
  %val, %idx = iree_gpu.arg_compare {1}(%input : vector<4x8xf32>)
                 inits(%init_val, %init_idx : vector<4xf32>, vector<4xi32>) {
  ^bb0(%lhs: f32, %rhs: f32):
    iree_gpu.yield %lhs : f32
  } -> vector<4xf32>, vector<4xi32>
  return
}

// -----

func.func @arg_compare_scalar_output_not_supported(%input: vector<8xf32>,
                                                    %init_val: f32,
                                                    %init_idx: i32) {
  %val, %idx = iree_gpu.arg_compare {0}(%input : vector<8xf32>)
                 // expected-error @+1 {{custom op 'iree_gpu.arg_compare' invalid kind of type specified}}
                 inits(%init_val, %init_idx : f32, i32) {
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = arith.cmpf ogt, %lhs, %rhs : f32
    iree_gpu.yield %cmp : i1
  } -> f32, i32
  return
}
