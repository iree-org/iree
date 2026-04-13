// RUN: iree-opt %s -iree-transform-dialect-interpreter -transform-dialect-drop-schedule --iree-gpu-test-target=gfx950 -o /dev/null 2>&1 | FileCheck %s

#contraction_accesses = [
 affine_map<() -> ()>,
 affine_map<() -> ()>,
 affine_map<() -> ()>
]
func.func @lower_vdmfma_f8_fnuz_on_gfx950(%A: vector<16xf8E4M3FNUZ>, %B: vector<32xf8E4M3FNUZ>, %C: vector<2xf32>) -> vector<2xf32> {
  %0 = iree_codegen.inner_tiled ins(%A, %B) outs(%C) {
    indexing_maps = #contraction_accesses,
    iterator_types = [],
    kind = #iree_gpu.virtual_mma_layout<VDMFMA_F32_8x16x128_F8E4M3FNUZ>,
    semantics = #iree_gpu.mma_semantics<distributed = true, opaque = false>
  } : vector<16xf8E4M3FNUZ>, vector<32xf8E4M3FNUZ> into vector<2xf32>
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

// CHECK: error: FNUZ fp8 VDMFMA virtual intrinsics are gfx942-only; use the IEEE fp8 VDMFMA variants on gfx950
