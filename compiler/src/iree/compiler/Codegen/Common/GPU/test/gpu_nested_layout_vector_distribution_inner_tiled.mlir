// RUN: iree-opt --iree-transform-dialect-interpreter --split-input-file --canonicalize --cse %s | FileCheck %s

// MFMA_F32_16x16x16_F16 with asymmetric outer iterations (i=2, j=4, k=3)

#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]

// LHS: shape = 2x3x16x16 (i, k)
#layout_lhs = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1, 1, 1],
  batch_tile    = [2, 3, 1, 1],
  outer_tile    = [1, 1, 1, 1],
  thread_tile   = [1, 1, 16, 4],
  element_tile  = [1, 1, 1, 4],

  subgroup_strides = [1, 1, 1, 1],
  thread_strides   = [0, 0, 1, 16]
>

// RHS: shape = 3x4x16x16 (k, j)
#layout_rhs = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1, 1, 1],
  batch_tile    = [3, 4, 1, 1],
  outer_tile    = [1, 1, 1, 1],
  thread_tile   = [1, 1, 4, 16],
  element_tile  = [1, 1, 4, 1],

  subgroup_strides = [1, 1, 1, 1],
  thread_strides   = [0, 0, 16, 1]
>

// ACC: shape = 2x4x16x16 (i, j)
#layout_acc = #iree_vector_ext.nested_layout<
  subgroup_tile = [1, 1, 1, 1],
  batch_tile    = [2, 4, 1, 1],
  outer_tile    = [1, 1, 1, 1],
  thread_tile   = [1, 1, 4, 16],
  element_tile  = [1, 1, 4, 1],

  subgroup_strides = [1, 1, 1, 1],
  thread_strides   = [0, 0, 16, 1]
>

// CHECK-LABEL: func @distribute_inner_tiled_mfma_16x16x16
// CHECK-SAME: (%[[LHS:.+]]: vector<2x3x16x16xf16>, %[[RHS:.+]]: vector<3x4x16x16xf16>, %[[ACC:.+]]: vector<2x4x16x16xf32>)
// CHECK-DAG: %[[LHS_SIMT:.+]] = iree_vector_ext.to_simt %[[LHS]] : vector<2x3x16x16xf16> -> vector<2x3x1x1x1x1x1x1x1x1x1x4xf16>
// CHECK-DAG: %[[RHS_SIMT:.+]] = iree_vector_ext.to_simt %[[RHS]] : vector<3x4x16x16xf16> -> vector<3x4x1x1x1x1x1x1x1x1x4x1xf16>
// CHECK-DAG: %[[ACC_SIMT:.+]] = iree_vector_ext.to_simt %[[ACC]] : vector<2x4x16x16xf32> -> vector<2x4x1x1x1x1x1x1x1x1x4x1xf32>
// CHECK-DAG: %[[LHS_CAST:.+]] = vector.shape_cast %[[LHS_SIMT]] : vector<2x3x1x1x1x1x1x1x1x1x1x4xf16> to vector<2x3x1x4xf16>
// CHECK-DAG: %[[RHS_CAST:.+]] = vector.shape_cast %[[RHS_SIMT]] : vector<3x4x1x1x1x1x1x1x1x1x4x1xf16> to vector<3x4x4x1xf16>
// CHECK-DAG: %[[ACC_CAST:.+]] = vector.shape_cast %[[ACC_SIMT]] : vector<2x4x1x1x1x1x1x1x1x1x4x1xf32> to vector<2x4x4x1xf32>
// CHECK:     %[[RESULT:.+]] = iree_codegen.inner_tiled ins(%[[LHS_CAST]], %[[RHS_CAST]]) outs(%[[ACC_CAST]])
// CHECK-SAME:   semantics = #iree_gpu.mma_semantics<distributed = true, opaque = true>
// CHECK-SAME:   : vector<2x3x1x4xf16>, vector<3x4x4x1xf16> into vector<2x4x4x1xf32>
// CHECK:     %[[RESULT_CAST:.+]] = vector.shape_cast %[[RESULT]] : vector<2x4x4x1xf32> to vector<2x4x1x1x1x1x1x1x1x1x4x1xf32>
// CHECK:     %[[RESULT_SIMD:.+]] = iree_vector_ext.to_simd %[[RESULT_CAST]] : vector<2x4x1x1x1x1x1x1x1x1x4x1xf32> -> vector<2x4x16x16xf32>
// CHECK:     return %[[RESULT_SIMD]]
func.func @distribute_inner_tiled_mfma_16x16x16(
    %lhs: vector<2x3x16x16xf16>,
    %rhs: vector<3x4x16x16xf16>,
    %acc: vector<2x4x16x16xf32>) -> vector<2x4x16x16xf32> {
  %A = iree_vector_ext.to_layout %lhs to layout(#layout_lhs) : vector<2x3x16x16xf16>
  %B = iree_vector_ext.to_layout %rhs to layout(#layout_rhs) : vector<3x4x16x16xf16>
  %C = iree_vector_ext.to_layout %acc to layout(#layout_acc) : vector<2x4x16x16xf32>

  %result = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = false, opaque = true>
  } : vector<2x3x16x16xf16>, vector<3x4x16x16xf16> into vector<2x4x16x16xf32>

  %O = iree_vector_ext.to_layout %result to layout(#layout_acc) : vector<2x4x16x16xf32>
  return %O : vector<2x4x16x16xf32>
}

builtin.module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.test_gpu_vector_distribution %top_level_func : !transform.any_op
    transform.yield
  }
}
