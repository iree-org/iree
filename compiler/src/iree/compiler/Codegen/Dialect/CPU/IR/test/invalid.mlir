// RUN: iree-opt --split-input-file --verify-diagnostics %s

// Checks invalid values for special key entries. We don't check the error
// message because they are IREE::Codegen::LoweringConfigTilingLevelAttr
// specific. We only care if an error is produced or not.

// expected-error@+1 {{}}
#invalid_empty_config = #iree_cpu.lowering_config<{}>

// -----

// expected-error@+1 {{}}
#invalid_distribution_config = #iree_cpu.lowering_config<distribution = 128 : i64>

// -----

// expected-error@+1 {{}}
#invalid_cache_parallel_config = #iree_cpu.lowering_config<cache_parallel = 128 : i64>

// -----

// expected-error@+1 {{}}
#invalid_cache_reduction_config = #iree_cpu.lowering_config<cache_reduction = 128 : i64>

// -----

// expected-error@+1 {{}}
#invalid_vector_common_parallel_config = #iree_cpu.lowering_config<vector_common_parallel = 128 : i64>

// -----

// expected-error@+1 {{}}
#invalid_vector_reduction_config = #iree_cpu.lowering_config<vector_reduction = 128 : i64>

// -----

// expected-error@+1 {{}}
#invalid_vector_inner_parallel_config = #iree_cpu.lowering_config<vector_inner_parallel = 128 : i64>

// -----

// `inner_tiled` rejects kind and semantics attributes from different dialects.

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
func.func @cpu_inner_tiled_requires_cpu_semantics(
    %lhs: tensor<1x1x16x1xf32>, %rhs: tensor<1x1x16x1xf32>, %acc: tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32> {
  // expected-error @+1 {{'iree_codegen.inner_tiled' op kind attribute (dialect 'iree_cpu') and semantics attribute (dialect 'iree_gpu') must use the same dialect namespace}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x1_F32_F32, intrinsics_m = 16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : tensor<1x1x16x1xf32>, tensor<1x1x16x1xf32> into tensor<1x1x16x16xf32>
  return %0 : tensor<1x1x16x16xf32>
}

// -----

// Inner tiles do not match the MMA tiles of the (default) 1×1×1 intrinsic:
// `intrinsics_m = 16` is needed to match the 16×1 / 16×1 / 16×16 inner shapes.

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
func.func @cpu_inner_tiled_f32_avx512_missing_intrinsics_m(
    %lhs: tensor<1x1x16x1xf32>, %rhs: tensor<1x1x16x1xf32>, %acc: tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32> {
  // expected-error @+1 {{'iree_codegen.inner_tiled' op operand #0 inner tile 'tensor<16x1xf32>' is incompatible with expected MMA tile type 'vector<1x1xf32>'}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x1_F32_F32>,
    semantics = #iree_cpu.mma_semantics<>
  } : tensor<1x1x16x1xf32>, tensor<1x1x16x1xf32> into tensor<1x1x16x16xf32>
  return %0 : tensor<1x1x16x16xf32>
}

// -----

// Operand element type must match the MMA tile element type.

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
func.func @cpu_inner_tiled_element_type_mismatch(
    %lhs: tensor<1x1x16x1xf16>, %rhs: tensor<1x1x16x1xf16>, %acc: tensor<1x1x16x16xf32>) -> tensor<1x1x16x16xf32> {
  // expected-error @+1 {{'iree_codegen.inner_tiled' op operand #0 inner tile 'tensor<16x1xf16>' is incompatible with expected MMA tile type 'vector<16x1xf32>'}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_X86_AVX512_1x16x1_F32_F32, intrinsics_m = 16>,
    semantics = #iree_cpu.mma_semantics<>
  } : tensor<1x1x16x1xf16>, tensor<1x1x16x1xf16> into tensor<1x1x16x16xf32>
  return %0 : tensor<1x1x16x16xf32>
}

// -----

// Scalable MMA tile axes (here N in `vector<1x[4]xf32>`) require a dynamic
// tensor extent; a fixed extent like 4 does not match.

#contraction_accesses = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
func.func @cpu_inner_tiled_sve_rhs_n_must_be_dynamic(
    %lhs: tensor<1x1x1x1xf32>, %rhs: tensor<1x1x4x1xf32>, %acc: tensor<1x1x1x?xf32>) -> tensor<1x1x1x?xf32> {
  // expected-error @+1 {{'iree_codegen.inner_tiled' op operand #1 inner tile 'tensor<4x1xf32>' is incompatible with expected MMA tile type 'vector<[4]x1xf32>'}}
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_ARM_SVE_FMLA_1x4VLx1_F32_F32>,
    semantics = #iree_cpu.mma_semantics<>
  } : tensor<1x1x1x1xf32>, tensor<1x1x4x1xf32> into tensor<1x1x1x?xf32>
  return %0 : tensor<1x1x1x?xf32>
}
