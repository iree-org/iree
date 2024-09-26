// RUN: iree-opt %s --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-verify-distribution))"

func.func @unmapped_forall(%out : memref<32xi32>) {
  // expected-error @+1 {{requires a mapping attribute}}
  scf.forall (%arg0) in (32) {
  }
  return
}

// -----

func.func @lane_forall_no_warp_parent(%out : memref<32xi32>) {
  // expected-error@+1 {{lane distributed scf.forall must have a parent subgroup distributed loop}}
  scf.forall (%arg0) in (32) {
  } {mapping = [#iree_gpu.lane_id<0>]}
  return
}
