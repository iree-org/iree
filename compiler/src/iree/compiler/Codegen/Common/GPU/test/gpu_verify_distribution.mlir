// RUN: iree-opt %s --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-verify-distribution))"

// expected-error @+1 {{requires a workgroup size attribute}}
func.func @incomplete_funcop(%out : memref<32xi32>) {
  scf.forall (%arg0) in (32) {
  }
  return
}

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
func.func @unmapped_forall(%out : memref<32xi32>) attributes {translation_info = #translation} {
  // expected-error @+1 {{requires a mapping attribute}}
  scf.forall (%arg0) in (32) {
  }
  return
}

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
func.func @lane_forall_no_warp_parent(%out : memref<32xi32>) attributes {translation_info = #translation} {
  // expected-error@+1 {{lane distributed scf.forall must have a parent subgroup distributed loop}}
  scf.forall (%arg0) in (32) {
  } {mapping = [#iree_gpu.lane_id<0>]}
  return
}
