// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file --iree-gpu-test-target=gfx942 | FileCheck %s --check-prefixes=CHECK,GFX942
// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --split-input-file --iree-gpu-test-target=gfx950 | FileCheck %s --check-prefixes=CHECK,GFX950

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_vdmfma_f16_target_sensitive(%A: vector<8xf16>, %B: vector<16xf16>, %C: vector<2xf32>) -> vector<2xf32> {
  %0 = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x64_F16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<8xf16>, vector<16xf16> into vector<2xf32>
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

// CHECK-LABEL: func @lower_vdmfma_f16_target_sensitive
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<8xf16>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<16xf16>
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<2xf32>
// CHECK: %[[ACC_EXPAND:.+]] = util.hoistable_conversion "vdmfma_interleave_acc"
// GFX942: %[[A_LO:.+]] = vector.extract_strided_slice %[[A]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// GFX942: %[[B_SLICE_0:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 8, 9, 2, 3, 10, 11] : vector<16xf16>, vector<16xf16>
// GFX942: %[[SMFMAC_0:.+]] = amdgpu.sparse_mfma 16x16x32 %[[A_LO]] * %[[B_SLICE_0]] + %[[ACC_EXPAND]]
// GFX942: %[[A_HI:.+]] = vector.extract_strided_slice %[[A]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
// GFX942: %[[B_SLICE_1:.+]] = vector.shuffle %[[B]], %[[B]] [4, 5, 12, 13, 6, 7, 14, 15] : vector<16xf16>, vector<16xf16>
// GFX942: %[[SMFMAC_1:.+]] = amdgpu.sparse_mfma 16x16x32 %[[A_HI]] * %[[B_SLICE_1]] + %[[SMFMAC_0]]
// GFX950: %[[B_WIDE:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15] : vector<16xf16>, vector<16xf16>
// GFX950: %[[SMFMAC:.+]] = amdgpu.sparse_mfma 16x16x64 %[[A]] * %[[B_WIDE]] + %[[ACC_EXPAND]]
// GFX950-NOT: amdgpu.sparse_mfma
// CHECK: %[[ACC_COLLAPSE:.+]] = util.hoistable_conversion "vdmfma_deinterleave_acc"

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_vdmfma_i8_target_sensitive(%A: vector<16xi8>, %B: vector<32xi8>, %C: vector<2xi32>) -> vector<2xi32> {
  %0 = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VDMFMA_I32_8x16x128_I8>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<16xi8>, vector<32xi8> into vector<2xi32>
  return %0 : vector<2xi32>
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

// CHECK-LABEL: func @lower_vdmfma_i8_target_sensitive
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<16xi8>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<32xi8>
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<2xi32>
// CHECK: %[[ACC_EXPAND:.+]] = util.hoistable_conversion "vdmfma_interleave_acc"
// GFX942: %[[A_LO:.+]] = vector.extract_strided_slice %[[A]] {offsets = [0], sizes = [8], strides = [1]} : vector<16xi8> to vector<8xi8>
// GFX942: %[[B_SLICE_0:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23] : vector<32xi8>, vector<32xi8>
// GFX942: %[[SMFMAC_0:.+]] = amdgpu.sparse_mfma 16x16x64 %[[A_LO]] * %[[B_SLICE_0]] + %[[ACC_EXPAND]]
// GFX942: %[[A_HI:.+]] = vector.extract_strided_slice %[[A]] {offsets = [8], sizes = [8], strides = [1]} : vector<16xi8> to vector<8xi8>
// GFX942: %[[B_SLICE_1:.+]] = vector.shuffle %[[B]], %[[B]] [8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31] : vector<32xi8>, vector<32xi8>
// GFX942: %[[SMFMAC_1:.+]] = amdgpu.sparse_mfma 16x16x64 %[[A_HI]] * %[[B_SLICE_1]] + %[[SMFMAC_0]]
// GFX950: %[[B_WIDE:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23, 8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31] : vector<32xi8>, vector<32xi8>
// GFX950: %[[SMFMAC:.+]] = amdgpu.sparse_mfma 16x16x128 %[[A]] * %[[B_WIDE]] + %[[ACC_EXPAND]]
// GFX950-NOT: amdgpu.sparse_mfma
// CHECK: %[[ACC_COLLAPSE:.+]] = util.hoistable_conversion "vdmfma_deinterleave_acc"

// -----

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_vdmfma_bf16_target_sensitive(%A: vector<8xbf16>, %B: vector<16xbf16>, %C: vector<2xf32>) -> vector<2xf32> {
  %0 = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x64_BF16>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<8xbf16>, vector<16xbf16> into vector<2xf32>
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

// CHECK-LABEL: func @lower_vdmfma_bf16_target_sensitive
// CHECK-SAME: %[[A:[A-Za-z0-9]+]]: vector<8xbf16>
// CHECK-SAME: %[[B:[A-Za-z0-9]+]]: vector<16xbf16>
// CHECK-SAME: %[[C:[A-Za-z0-9]+]]: vector<2xf32>
// CHECK: %[[ACC_EXPAND:.+]] = util.hoistable_conversion "vdmfma_interleave_acc"
// GFX942: %[[A_LO:.+]] = vector.extract_strided_slice %[[A]] {offsets = [0], sizes = [4], strides = [1]} : vector<8xbf16> to vector<4xbf16>
// GFX942: %[[B_SLICE_0:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 8, 9, 2, 3, 10, 11] : vector<16xbf16>, vector<16xbf16>
// GFX942: %[[SMFMAC_0:.+]] = amdgpu.sparse_mfma 16x16x32 %[[A_LO]] * %[[B_SLICE_0]] + %[[ACC_EXPAND]]
// GFX942: %[[A_HI:.+]] = vector.extract_strided_slice %[[A]] {offsets = [4], sizes = [4], strides = [1]} : vector<8xbf16> to vector<4xbf16>
// GFX942: %[[B_SLICE_1:.+]] = vector.shuffle %[[B]], %[[B]] [4, 5, 12, 13, 6, 7, 14, 15] : vector<16xbf16>, vector<16xbf16>
// GFX942: %[[SMFMAC_1:.+]] = amdgpu.sparse_mfma 16x16x32 %[[A_HI]] * %[[B_SLICE_1]] + %[[SMFMAC_0]]
// GFX950: %[[B_WIDE:.+]] = vector.shuffle %[[B]], %[[B]] [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15] : vector<16xbf16>, vector<16xbf16>
// GFX950: %[[SMFMAC:.+]] = amdgpu.sparse_mfma 16x16x64 %[[A]] * %[[B_WIDE]] + %[[ACC_EXPAND]]
// GFX950-NOT: amdgpu.sparse_mfma
// CHECK: %[[ACC_COLLAPSE:.+]] = util.hoistable_conversion "vdmfma_deinterleave_acc"
