// RUN: iree-opt %s --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-verify-distribution))"

// expected-error @+1 {{requires a workgroup size attribute}}
func.func @incomplete_funcop() {
  scf.forall (%arg0) in (32) {
  }
  return
}

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
func.func @unmapped_forall() attributes {translation_info = #translation} {
  // expected-error @+1 {{requires a mapping attribute}}
  scf.forall (%arg0) in (32) {
  }
  return
}

// -----

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
func.func @lane_forall_no_warp_parent() attributes {translation_info = #translation} {
  // expected-error @+1 {{lane distributed scf.forall must have a parent subgroup distributed loop}}
  scf.forall (%arg0) in (32) {
  } {mapping = [#iree_gpu.lane_id<0>]}
  return
}

// -----

// Writes inside thread-mapped foralls should pass verification.

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
func.func @thread_forall_write_ok(%buf: memref<64xf32, #gpu.address_space<workgroup>>)
    attributes {translation_info = #translation} {
  scf.forall (%tid) in (64) {
    %cst = arith.constant 0.0 : f32
    memref.store %cst, %buf[%tid] : memref<64xf32, #gpu.address_space<workgroup>>
  } {mapping = [#gpu.thread<x>]}
  return
}

// -----

// Writes outside any distributed context should fail.

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 32>
func.func @undistributed_write(%buf: memref<64xf32, #gpu.address_space<workgroup>>)
    attributes {translation_info = #translation} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  // expected-error @+1 {{write affecting operations on shared resources are restricted to lane or thread distributed contexts}}
  memref.store %cst, %buf[%c0] : memref<64xf32, #gpu.address_space<workgroup>>
  return
}

// -----

// Writes inside pcf.generic should pass verification.

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64>
func.func @pcf_generic_write_ok(%buf: memref<256xf16, #gpu.address_space<workgroup>>)
    attributes {translation_info = #translation} {
  pcf.generic scope(#iree_gpu.subgroup_scope)
    execute[%sg_id: index, %num_sg: index] {
    pcf.generic scope(#iree_gpu.lane_scope)
      execute[%lane_id: index, %sg_size: index] {
      %cst = arith.constant 0.0 : f16
      memref.store %cst, %buf[%lane_id] : memref<256xf16, #gpu.address_space<workgroup>>
      pcf.return
    }
    pcf.return
  }
  return
}

// -----

// Writes inside lane-scoped pcf.loop should pass verification.

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64>
func.func @pcf_loop_lane_scope_write_ok(%buf: memref<256xf16, #gpu.address_space<workgroup>>)
    attributes {translation_info = #translation} {
  %c4 = arith.constant 4 : index
  pcf.loop scope(#iree_gpu.lane_scope) count(%c4)
    execute[%iv: index] {
    %cst = arith.constant 0.0 : f16
    memref.store %cst, %buf[%iv] : memref<256xf16, #gpu.address_space<workgroup>>
    pcf.return
  }
  return
}

// -----

// Writes inside subgroup-scoped pcf.generic (without lane nesting) should fail.

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64>
func.func @pcf_subgroup_scope_write_fail(%buf: memref<256xf16, #gpu.address_space<workgroup>>)
    attributes {translation_info = #translation} {
  pcf.generic scope(#iree_gpu.subgroup_scope)
    execute[%sg_id: index, %num_sg: index] {
    %cst = arith.constant 0.0 : f16
    // expected-error @+1 {{write affecting operations on shared resources are restricted to lane or thread distributed contexts}}
    memref.store %cst, %buf[%sg_id] : memref<256xf16, #gpu.address_space<workgroup>>
    pcf.return
  }
  return
}

// -----

// Write outside pcf.generic (but pcf.generic exists elsewhere) should fail.

#translation = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [512, 1, 1] subgroup_size = 64>
func.func @write_outside_pcf_generic(%buf: memref<256xf16, #gpu.address_space<workgroup>>)
    attributes {translation_info = #translation} {
  pcf.generic scope(#iree_gpu.subgroup_scope)
    execute[%sg_id: index, %num_sg: index] {
    pcf.return
  }
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  // expected-error @+1 {{write affecting operations on shared resources are restricted to lane or thread distributed contexts}}
  memref.store %cst, %buf[%c0] : memref<256xf16, #gpu.address_space<workgroup>>
  return
}
