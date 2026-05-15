// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 --iree-convert-to-rocdl %s | FileCheck %s

// Test permlane lowering on gfx950.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
module {
  func.func @test_permlane_16_32_lowering() {
    %c0  = arith.constant 0 : index
    %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<1xi32>

    %tid = gpu.thread_id x
    %val = arith.index_castui %tid : index to i32

    // Emits rocdl.permlane*.swap on gfx950.
    %p32 = amdgpu.permlane_swap %val 32 : i32
    %a32 = arith.addi %val, %p32 : i32
    %p16 = amdgpu.permlane_swap %a32 16 : i32
    %sum = arith.addi %a32, %p16 : i32

    %is0 = arith.cmpi eq, %tid, %c0 : index
    scf.if %is0 {
      memref.store %sum, %out[%c0] : memref<1xi32>
    }
    return
  }
}

// CHECK-LABEL: llvm.func @test_permlane_16_32_lowering
// CHECK: rocdl.permlane32.swap
// CHECK: rocdl.permlane16.swap

// -----

module {
  func.func @global_subgroup_barrier() {
    iree_gpu.global_subgroup_barrier
    return
  }
}

// CHECK-LABEL: llvm.func @global_subgroup_barrier
//       CHECK:   rocdl.s.barrier

// -----

// Verify that arith.truncf f32 to bf16 is NOT expanded on gfx950, which has
// native bf16 conversion instructions (v_cvt_pk_bf16_f32).
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
module {
  func.func @bf16_truncf_native() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<64xf32>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<64xbf16>
    %val = memref.load %0[%c0] : memref<64xf32>
    %trunc = arith.truncf %val : f32 to bf16
    memref.store %trunc, %1[%c0] : memref<64xbf16>
    return
  }
}
// CHECK-LABEL: llvm.func @bf16_truncf_native
//       CHECK:   llvm.fptrunc
//   CHECK-NOT:   llvm.lshr
//       CHECK:   llvm.return
