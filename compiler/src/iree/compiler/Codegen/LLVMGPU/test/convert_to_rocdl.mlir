// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-convert-to-rocdl))))" %s | FileCheck %s

// Test that that standard and GPU ops are converted to LLVM and NVVM.
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>,
  #hal.descriptor_set.layout<1, bindings = [
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @abs_ex_dispatch_0 {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @abs_ex_dispatch_0 layout(#pipeline_layout)
    builtin.module {
      func.func @abs_ex_dispatch_0() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) flags(ReadOnly) : memref<16xf32>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16xf32>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<16xf32>
        %3 = gpu.block_id x
        %4 = gpu.block_dim x
        %5 = gpu.thread_id x
        %6 = arith.muli %3, %4 : index
        %7 = arith.addi %6, %5 : index
        %9 = memref.load %1[%7] : memref<16xf32>
        %10 = memref.load %2[%7] : memref<16xf32>
        %11 = arith.addf %9, %10 : f32
        memref.store %11, %0[%7] : memref<16xf32>
        return
      }
    }
  }
}
// CHECK-LABEL: llvm.func @abs_ex_dispatch_0
//  CHECK-SAME: (%{{[a-zA-Z0-9]*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.readonly},
//  CHECK-SAME:  %{{[a-zA-Z0-9]*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias},
//  CHECK-SAME:  %{{[a-zA-Z0-9]*}}: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias})
//      CHECK:    rocdl.workgroup.dim.x
//      CHECK:    llvm.fadd


// -----
// Test that maximum and minum are converted to max and min on rocm
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>,
  #hal.descriptor_set.layout<1, bindings = [
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable @abs_ex_dispatch_0 {
  hal.executable.variant @cuda target(<"cuda", "cuda-nvptx-fb">) {
    hal.executable.export @abs_ex_dispatch_0 layout(#pipeline_layout)
    builtin.module {
      func.func @reduction_maximum() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) :
            memref<32x64x64xf32, strided<[4096, 64, 1], offset: ?>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<32x64x64xf32,
            strided<[4096, 64, 1], offset: ?>>
      %2 = vector.load %0[%c0, %c0, %c0] : memref<32x64x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<2xf32>
      %3 = vector.reduction <maximumf>, %2 : vector<2xf32> into f32
      %4 = vector.splat %3 : vector<2xf32>
      vector.store %4, %1[%c0, %c0, %c0] : memref<32x64x64xf32, strided<[4096, 64, 1], offset: ?>>, vector<2xf32>
      return
      }
    }
  }
}
// CHECK-LABEL: llvm.func @reduction_maximum
// CHECK:  llvm.intr.vector.reduce.fmax({{.*}})  : (vector<2xf32>) -> f32

// -----
// Test that gpu barriers be lowered to `s_waitcnt lgkmcnt(0)\0As_barrier` on rocm
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>
  ]>
]>
hal.executable @simple_barrier {
  hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export @simple_barrier layout(#pipeline_layout)
    builtin.module {
      func.func @simple_barrier() {
        gpu.barrier
        return
      }
    }
  }
}
// CHECK-LABEL: llvm.func @simple_barrier
// CHECK: llvm.inline_asm has_side_effects asm_dialect = att "s_waitcnt lgkmcnt(0)\0As_barrier", ""  : () -> ()

// -----
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @masked_load_store {
  hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export @masked_load_store layout(#pipeline_layout)
    builtin.module {
      func.func @masked_load_store() {
        %c0 = arith.constant 0 : index
        %idx = gpu.thread_id x
        %pass_thru = arith.constant dense<0.000000e+00> : vector<1xf32>
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<64xf32, #gpu.address_space<global>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<64xf32, #gpu.address_space<global>>
        %mask = vector.create_mask %idx : vector<1xi1>
        %ld = vector.maskedload %0[%idx], %mask, %pass_thru : memref<64xf32, #gpu.address_space<global>>, vector<1xi1>, vector<1xf32> into vector<1xf32>
        vector.maskedstore %1[%idx], %mask, %ld : memref<64xf32, #gpu.address_space<global>>, vector<1xi1>, vector<1xf32>
        return
      }
    }
  }
}
// CHECK-LABEL: llvm.func @masked_load_store
//       CHECK:   %[[MASK_BIT:.+]] = llvm.icmp "sgt" {{.*}} : vector<1xi64>
//       CHECK:   llvm.intr.masked.load %{{.*}}, %[[MASK_BIT]]
//       CHECK:   llvm.intr.masked.store %{{.*}}, %[[MASK_BIT]]
