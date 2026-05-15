// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --iree-convert-to-rocdl %s | FileCheck %s

// Verify that arith.truncf f32 to bf16 is not expanded on gfx942. The LLVM
// backend handles bf16 emulation during instruction selection for targets
// lacking native bf16 conversion instructions (v_cvt_pk_bf16_f32).
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

// -----

// Verify that arith.extf bf16 to f32 is not expanded on gfx942.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
module {
  func.func @bf16_extf_native() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<64xbf16>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : memref<64xf32>
    %val = memref.load %0[%c0] : memref<64xbf16>
    %ext = arith.extf %val : bf16 to f32
    memref.store %ext, %1[%c0] : memref<64xf32>
    return
  }
}
// CHECK-LABEL: llvm.func @bf16_extf_native
//       CHECK:   llvm.fpext
//   CHECK-NOT:   llvm.shl
//       CHECK:   llvm.return
