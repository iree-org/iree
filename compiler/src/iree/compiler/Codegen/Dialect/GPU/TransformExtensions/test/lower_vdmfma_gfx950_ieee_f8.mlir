// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file --iree-gpu-test-target=gfx950 | FileCheck %s

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_vdmfma_f8e4m3fn_gfx950(%A: vector<16xf8E4M3FN>, %B: vector<32xf8E4M3FN>, %C: vector<2xf32>) -> vector<2xf32> {
  %0 = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x128_F8E4M3FN>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<16xf8E4M3FN>, vector<32xf8E4M3FN> into vector<2xf32>
  return %0 : vector<2xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_vdmfma_f8e4m3fn_gfx950
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xf8E4M3FN>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<32xf8E4M3FN>
// CHECK: %[[ACC_EXPAND:.+]] = util.hoistable_conversion "vdmfma_interleave_acc"
// CHECK: %[[B_WIDE:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23, 8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31] : vector<32xf8E4M3FN>, vector<32xf8E4M3FN>
// CHECK: amdgpu.sparse_mfma 16x16x128 %[[A]] * %[[B_WIDE]] + %[[ACC_EXPAND]] sparse(%{{.+}} : vector<2xi16>) : vector<16xf8E4M3FN>, vector<32xf8E4M3FN>, vector<4xf32>
// CHECK-NOT: amdgpu.sparse_mfma

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_vdmfma_f8e5m2_gfx950(%A: vector<16xf8E5M2>, %B: vector<32xf8E5M2>, %C: vector<2xf32>) -> vector<2xf32> {
  %0 = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x128_F8E5M2>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<16xf8E5M2>, vector<32xf8E5M2> into vector<2xf32>
  return %0 : vector<2xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_vdmfma_f8e5m2_gfx950
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xf8E5M2>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<32xf8E5M2>
// CHECK: %[[ACC_EXPAND:.+]] = util.hoistable_conversion "vdmfma_interleave_acc"
// CHECK: %[[B_WIDE:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23, 8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31] : vector<32xf8E5M2>, vector<32xf8E5M2>
// CHECK: amdgpu.sparse_mfma 16x16x128 %[[A]] * %[[B_WIDE]] + %[[ACC_EXPAND]] sparse(%{{.+}} : vector<2xi16>) : vector<16xf8E5M2>, vector<32xf8E5M2>, vector<4xf32>
// CHECK-NOT: amdgpu.sparse_mfma

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_vdmfma_f8e5m2_f8e4m3fn_gfx950(%A: vector<16xf8E5M2>, %B: vector<32xf8E4M3FN>, %C: vector<2xf32>) -> vector<2xf32> {
  %0 = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x128_F8E5M2_F8E4M3FN>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<16xf8E5M2>, vector<32xf8E4M3FN> into vector<2xf32>
  return %0 : vector<2xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_vdmfma_f8e5m2_f8e4m3fn_gfx950
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xf8E5M2>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<32xf8E4M3FN>
// CHECK: %[[ACC_EXPAND:.+]] = util.hoistable_conversion "vdmfma_interleave_acc"
// CHECK: %[[B_WIDE:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23, 8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31] : vector<32xf8E4M3FN>, vector<32xf8E4M3FN>
// CHECK: amdgpu.sparse_mfma 16x16x128 %[[A]] * %[[B_WIDE]] + %[[ACC_EXPAND]] sparse(%{{.+}} : vector<2xi16>) : vector<16xf8E5M2>, vector<32xf8E4M3FN>, vector<4xf32>
// CHECK-NOT: amdgpu.sparse_mfma

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_vdmfma_f8e4m3fn_f8e5m2_gfx950(%A: vector<16xf8E4M3FN>, %B: vector<32xf8E5M2>, %C: vector<2xf32>) -> vector<2xf32> {
  %0 = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x128_F8E4M3FN_F8E5M2>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<16xf8E4M3FN>, vector<32xf8E5M2> into vector<2xf32>
  return %0 : vector<2xf32>
}

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.lower_inner_tiled
    } : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL: func @lower_vdmfma_f8e4m3fn_f8e5m2_gfx950
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xf8E4M3FN>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<32xf8E5M2>
// CHECK: %[[ACC_EXPAND:.+]] = util.hoistable_conversion "vdmfma_interleave_acc"
// CHECK: %[[B_WIDE:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23, 8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31] : vector<32xf8E5M2>, vector<32xf8E5M2>
// CHECK: amdgpu.sparse_mfma 16x16x128 %[[A]] * %[[B_WIDE]] + %[[ACC_EXPAND]] sparse(%{{.+}} : vector<2xi16>) : vector<16xf8E4M3FN>, vector<32xf8E5M2>, vector<4xf32>
// CHECK-NOT: amdgpu.sparse_mfma
