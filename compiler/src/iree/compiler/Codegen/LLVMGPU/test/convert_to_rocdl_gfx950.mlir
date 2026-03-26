// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 --iree-convert-to-rocdl %s | FileCheck --check-prefix=CHECK-PERMLANE %s
// Targets with native bf16 conversion instructions (hasBF16ConversionInsts):
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950  --iree-convert-to-rocdl %s | FileCheck --check-prefix=CHECK-NATIVE-BF16 %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1250 --iree-convert-to-rocdl %s | FileCheck --check-prefix=CHECK-NATIVE-BF16 %s
// Targets without native bf16 conversion (software expansion via bitshifts):
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --iree-convert-to-rocdl %s | FileCheck --check-prefix=CHECK-NO-NATIVE-BF16 %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx90a --iree-convert-to-rocdl %s | FileCheck --check-prefix=CHECK-NO-NATIVE-BF16 %s

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

// CHECK-PERMLANE-LABEL: llvm.func @test_permlane_16_32_lowering
// CHECK-PERMLANE: rocdl.permlane32.swap
// CHECK-PERMLANE: rocdl.permlane16.swap

// -----

module {
  func.func @global_subgroup_barrier() {
    iree_gpu.global_subgroup_barrier
    return
  }
}

// CHECK-PERMLANE-LABEL: llvm.func @global_subgroup_barrier
//       CHECK-PERMLANE:   rocdl.s.barrier

// -----

// Test that targets with native bf16 conversion instructions (gfx950, gfx1250)
// skip software bf16 expansion, while targets without it (gfx942, gfx90a)
// expand via a bitshift sequence.
#pipeline_layout_bf16 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
module {
  func.func @test_bf16_truncf() {
    %c0 = arith.constant 0 : index
    %in  = hal.interface.binding.subspan layout(#pipeline_layout_bf16) binding(0) alignment(64) offset(%c0) : memref<4xf32>
    %out = hal.interface.binding.subspan layout(#pipeline_layout_bf16) binding(1) alignment(64) offset(%c0) : memref<4xbf16>
    %tid = gpu.thread_id x
    %val = memref.load %in[%tid] : memref<4xf32>
    %tr  = arith.truncf %val : f32 to bf16
    memref.store %tr, %out[%tid] : memref<4xbf16>
    return
  }
}

// On targets with native bf16 conversion the truncf is lowered directly
// without a software bitshift sequence.
// CHECK-NATIVE-BF16-LABEL: llvm.func @test_bf16_truncf
// CHECK-NATIVE-BF16-NOT: llvm.lshr
// CHECK-NATIVE-BF16-NOT: llvm.and

// On targets without native bf16 conversion the truncf is expanded via
// a software bitshift sequence.
// CHECK-NO-NATIVE-BF16-LABEL: llvm.func @test_bf16_truncf
// CHECK-NO-NATIVE-BF16: llvm.lshr
