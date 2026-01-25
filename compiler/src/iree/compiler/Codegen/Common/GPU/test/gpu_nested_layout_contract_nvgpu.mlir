// RUN: iree-opt --split-input-file --iree-transform-dialect-interpreter --canonicalize --cse %s | FileCheck %s

// NV_MMA_SYNC_F32_16x8x16_F16: M=16, N=8, K=16, subgroup_size=32
// From IREEGPUAttrs.cpp getSingleSubgroupLayout:
//   LHS: outer=[2,2], thread=[8,4], strides=[4,1], element=[1,2]
//   RHS: outer=[2,1], thread=[4,8], strides=[1,4], element=[2,1]
//   ACC: outer=[2,1], thread=[8,4], strides=[4,1], element=[1,2]

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

// A: shape = 16x16 (MxK), layout matches LHS
#layout_a = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile    = [2, 2],
  thread_tile   = [8, 4],
  element_tile  = [1, 2],

  subgroup_strides = [1, 1],
  thread_strides   = [4, 1]
>

// B: shape = 16x8 (KxN), layout matches RHS
#layout_b = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile    = [2, 1],
  thread_tile   = [4, 8],
  element_tile  = [2, 1],

  subgroup_strides = [1, 1],
  thread_strides   = [1, 4]
>

// C: shape = 16x8 (MxN), layout matches ACC
#layout_c = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile    = [2, 1],
  thread_tile   = [8, 4],
  element_tile  = [1, 2],

  subgroup_strides = [1, 1],
  thread_strides   = [4, 1]
>

func.func @contract_to_mma_sync_16x8x16_mm(%a : vector<16x16xf16>, %b : vector<16x8xf16>, %c : vector<16x8xf32>) -> vector<16x8xf32> {
  %A = iree_vector_ext.to_layout %a to layout(#layout_a) : vector<16x16xf16>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b) : vector<16x8xf16>
  %C = iree_vector_ext.to_layout %c to layout(#layout_c) : vector<16x8xf32>

  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.gpu.mma = #iree_gpu.mma_layout<NV_MMA_SYNC_F32_16x8x16_F16>
  } %A, %B, %C : vector<16x16xf16>, vector<16x8xf16> into vector<16x8xf32>

  %O = iree_vector_ext.to_layout %output to layout(#layout_c) : vector<16x8xf32>
  return %O : vector<16x8xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @contract_to_mma_sync_16x8x16_mm
// CHECK-SAME: (%[[A:.+]]: vector<16x16xf16>, %[[B:.+]]: vector<16x8xf16>, %[[C:.+]]: vector<16x8xf32>)
// CHECK:       %[[C_SIMT:.+]] = iree_vector_ext.to_simt %[[C]] : vector<16x8xf32>
// CHECK:       %[[A_SIMT:.+]] = iree_vector_ext.to_simt %[[A]] : vector<16x16xf16>
// CHECK:       %[[B_SIMT:.+]] = iree_vector_ext.to_simt %[[B]] : vector<16x8xf16>
// Verify LHS transpose sequence for mma.sync column-major ordering
// VectorDistribute produces 1x1x2x2x1x2 -> reshape to 2x2x2 -> transpose [1,0,2] -> reshape to 4x2
// CHECK:       %[[LHS_RESHAPE:.+]] = vector.shape_cast {{.*}} : vector<1x1x2x2x1x2xf16> to vector<2x2x2xf16>
// CHECK:       %[[LHS_TRANSPOSE:.+]] = vector.transpose %[[LHS_RESHAPE]], [1, 0, 2] : vector<2x2x2xf16> to vector<2x2x2xf16>
// CHECK:       %[[LHS_FINAL:.+]] = vector.shape_cast %[[LHS_TRANSPOSE]] : vector<2x2x2xf16> to vector<4x2xf16>
// Verify nvgpu.mma.sync is generated with correct shape and transposed LHS
// CHECK:       nvgpu.mma.sync(%[[LHS_FINAL]], {{.*}}) {mmaShape = [16, 8, 16]}
// CHECK:       iree_vector_ext.to_simd {{.*}} : {{.*}} -> vector<16x8xf32>

// -----

// Test with F16 accumulator: NV_MMA_SYNC_F16_16x8x16_F16

#map1 = affine_map<(m, n, k) -> (m, k)>
#map2 = affine_map<(m, n, k) -> (k, n)>
#map3 = affine_map<(m, n, k) -> (m, n)>

#layout_a_f16 = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile    = [2, 2],
  thread_tile   = [8, 4],
  element_tile  = [1, 2],

  subgroup_strides = [1, 1],
  thread_strides   = [4, 1]
>

#layout_b_f16 = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile    = [2, 1],
  thread_tile   = [4, 8],
  element_tile  = [2, 1],

  subgroup_strides = [1, 1],
  thread_strides   = [1, 4]
>

#layout_c_f16 = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1],
  batch_tile    = [1, 1],
  outer_tile    = [2, 1],
  thread_tile   = [8, 4],
  element_tile  = [1, 2],

  subgroup_strides = [1, 1],
  thread_strides   = [4, 1]
>

func.func @contract_to_mma_sync_16x8x16_f16_mm(%a : vector<16x16xf16>, %b : vector<16x8xf16>, %c : vector<16x8xf16>) -> vector<16x8xf16> {
  %A = iree_vector_ext.to_layout %a to layout(#layout_a_f16) : vector<16x16xf16>
  %B = iree_vector_ext.to_layout %b to layout(#layout_b_f16) : vector<16x8xf16>
  %C = iree_vector_ext.to_layout %c to layout(#layout_c_f16) : vector<16x8xf16>

  %output = vector.contract {
    indexing_maps = [#map1, #map2, #map3],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>,
    iree.gpu.mma = #iree_gpu.mma_layout<NV_MMA_SYNC_F16_16x8x16_F16>
  } %A, %B, %C : vector<16x16xf16>, vector<16x8xf16> into vector<16x8xf16>

  %O = iree_vector_ext.to_layout %output to layout(#layout_c_f16) : vector<16x8xf16>
  return %O : vector<16x8xf16>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @contract_to_mma_sync_16x8x16_f16_mm
// CHECK-SAME: (%[[A:.+]]: vector<16x16xf16>, %[[B:.+]]: vector<16x8xf16>, %[[C:.+]]: vector<16x8xf16>)
// CHECK:       %[[C_SIMT:.+]] = iree_vector_ext.to_simt %[[C]] : vector<16x8xf16>
// CHECK:       %[[A_SIMT:.+]] = iree_vector_ext.to_simt %[[A]] : vector<16x16xf16>
// CHECK:       %[[B_SIMT:.+]] = iree_vector_ext.to_simt %[[B]] : vector<16x8xf16>
// Verify LHS transpose sequence for mma.sync column-major ordering
// CHECK:       %[[LHS_RESHAPE:.+]] = vector.shape_cast {{.*}} : vector<1x1x2x2x1x2xf16> to vector<2x2x2xf16>
// CHECK:       %[[LHS_TRANSPOSE:.+]] = vector.transpose %[[LHS_RESHAPE]], [1, 0, 2] : vector<2x2x2xf16> to vector<2x2x2xf16>
// CHECK:       %[[LHS_FINAL:.+]] = vector.shape_cast %[[LHS_TRANSPOSE]] : vector<2x2x2xf16> to vector<4x2xf16>
// Verify nvgpu.mma.sync is generated with correct shape, transposed LHS, and f16 output
// CHECK:       nvgpu.mma.sync(%[[LHS_FINAL]], {{.*}}) {mmaShape = [16, 8, 16]} : ({{.*}}) -> vector<2x2xf16>
// CHECK:       iree_vector_ext.to_simd {{.*}} : {{.*}} -> vector<16x8xf16>
