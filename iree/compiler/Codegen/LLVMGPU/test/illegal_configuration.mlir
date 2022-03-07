// RUN: iree-opt -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target-pass{test-lowering-configuration=true}))' -verify-diagnostics -split-input-file %s

#config = #iree_codegen.lowering.config<tile_sizes = [], native_vector_size = []>
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulSimt", workload_per_wg = []>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) {
      translation.info = #translation,
      workgroup_size = [32 : index, 8 : index, 8 : index]
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8x16xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<4x16xf32>
        // expected-error @+1 {{expected workgroup size to be <=1024 for LLVMGPUMatmulSimt, got 2048}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [], native_vector_size = []>
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulSimt", workload_per_wg = []>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) {
      translation.info = #translation,
      workgroup_size = [32 : index, 8 : index, 2 : index]
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8x16xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<4x16xf32>
        // expected-error @+1 {{expected workgroup z component to be 1 for LLVMGPUMatmulSimt, got 2}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [[32, 32, 16]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulTensorCore", workload_per_wg = []>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) {
      translation.info = #translation,
      workgroup_size = [64 : index, 2 : index, 10 : index]
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<32x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x32xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<32x32xf32>
        // expected-error @+1 {{expected workgroup size to be <=1024 for LLVMGPUMatmulTensorCore, got 1280}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<32x16xf32>, memref<16x32xf32>)
          outs(%result: memref<32x32xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [[32, 32, 16]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulTensorCore", workload_per_wg = []>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) {
      translation.info = #translation,
      workgroup_size = [48 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<32x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x32xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<32x32xf32>
        // expected-error @+1 {{workgroup size is not 32 aligned for LLVMGPUMatmulTensorCore, got 48}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<32x16xf32>, memref<16x32xf32>)
          outs(%result: memref<32x32xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [[32, 32, 16]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulTensorCore", workload_per_wg = []>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) {
      translation.info = #translation,
      workgroup_size = [64 : index, 2 : index, 2 : index]
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<32x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x32xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<32x32xf32>
        // expected-error @+1 {{expected workgroup z component to be 1 for LLVMGPUMatmulTensorCore, got 2}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<32x16xf32>, memref<16x32xf32>)
          outs(%result: memref<32x32xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [[32, 32, 20]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulTensorCore", workload_per_wg = []>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) {
      translation.info = #translation,
      workgroup_size = [64 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<32x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x32xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<32x32xf32>
        // expected-error @+1 {{tensorcore size doesn't factor into second level tile size for LLVMGPUMatmulTensorCore}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<32x16xf32>, memref<16x32xf32>)
          outs(%result: memref<32x32xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [[32, 32, 16]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulTensorCore", workload_per_wg = []>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) {
      translation.info = #translation,
      workgroup_size = [64 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<48x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x32xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<48x32xf32>
        // expected-error @+1 {{lhsShape doesn't factor into first level tile size for LLVMGPUMatmulTensorCore}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<48x16xf32>, memref<16x32xf32>)
          outs(%result: memref<48x32xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [[32, 32, 16]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulTensorCore", workload_per_wg = []>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) {
      translation.info = #translation,
      workgroup_size = [64 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<32x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x48xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<32x48xf32>
        // expected-error @+1 {{rhsShape doesn't factor into first level tile size for LLVMGPUMatmulTensorCore}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<32x16xf32>, memref<16x48xf32>)
          outs(%result: memref<32x48xf32>)
        return
      }
    }
  }
}

// -----

// This test might not be valid anymore when setting configuration on untiled ops.

// #config = #iree_codegen.lowering.config<tile_sizes = [[2, 32, 32, 16]], native_vector_size = []>
// #translation = #iree_codegen.translation.info<"LLVMGPUMatmulTensorCore", workload_per_wg = []>
// #executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
// #executable_layout = #hal.executable.layout<push_constants = 0, sets = [
//   #hal.descriptor_set.layout<0, bindings = [
//     #hal.descriptor_set.binding<0, storage_buffer>,
//     #hal.descriptor_set.binding<1, storage_buffer>,
//     #hal.descriptor_set.binding<2, storage_buffer>
//   ]>
// ]>
// hal.executable private @batch_matmul_func  {
//   hal.executable.variant @cuda, target = #executable_target_cuda_nvptx_fb {
//     hal.executable.entry_point @batch_matmul_func layout(#executable_layout) {
//       translation.info = #translation,
//       workgroup_size = [64 : index, 2 : index, 1 : index]
//     }
// builtin.module {
//   func @batch_matmul_func() {
//     %c0 = arith.constant 0 : index
//     %cst = arith.constant 0.000000e+00 : f32
//     %c4 = arith.constant 4 : index
//     %c32 = arith.constant 32 : index
//     %c64 = arith.constant 64 : index
//     %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(32) : memref<4x32x1024xf32>
//     memref.assume_alignment %0, 32 : memref<4x32x1024xf32>
//     %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(32) : memref<4x1024x64xf32>
//     memref.assume_alignment %1, 32 : memref<4x1024x64xf32>
//     %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(32) : memref<4x32x64xf32>
//     memref.assume_alignment %2, 32 : memref<4x32x64xf32>
//     linalg.fill(%cst, %2) {lowering.config = #config} : f32, memref<4x32x64xf32>
//     // exp- ected-error @+1 {{Received first tile dimension of 2 instead of 1 for LLVMGPUMatmulTensorCore}}
//     linalg.batch_matmul {lowering.config = #config} ins(%0, %1 : memref<4x32x1024xf32>, memref<4x1024x64xf32>) outs(%2 : memref<4x32x64xf32>)
//     return
//   }
// }
// }
// }

// -----
