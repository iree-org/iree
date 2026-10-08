// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1201 --iree-convert-to-rocdl %s | FileCheck %s

module {
  func.func @global_subgroup_barrier() {
    iree_gpu.global_subgroup_barrier
    return
  }
}

// CHECK-LABEL: llvm.func @global_subgroup_barrier
//       CHECK:   rocdl.s.barrier.signal id = -1
//       CHECK:   rocdl.s.barrier.wait id = -1

// -----

// gfx1201 supports both wave32 and wave64. Without an explicit subgroup size
// the target's preferred size (wave32) is used, which shows up as the lower
// bound of the wavefront size range.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
module {
  func.func @subgroup_size_default() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<1xindex>
    %1 = gpu.subgroup_size upper_bound 64 : index
    memref.store %1, %0[%c0] : memref<1xindex>
    return
  }
}

// CHECK-LABEL: llvm.func @subgroup_size_default
//       CHECK:   rocdl.wavefrontsize range <i32, 32, 65> : i32

// -----

// An explicit wave64 subgroup size is threaded into the target description.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<VectorDistribute>
                                              workgroup_size = [64, 1, 1]
                                              subgroup_size = 64>
module {
  func.func @subgroup_size_wave64() attributes {translation_info = #translation} {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<1xindex>
    %1 = gpu.subgroup_size upper_bound 64 : index
    memref.store %1, %0[%c0] : memref<1xindex>
    return
  }
}

// CHECK-LABEL: llvm.func @subgroup_size_wave64
//       CHECK:   rocdl.wavefrontsize range <i32, 64, 65> : i32
