// RUN: iree-opt -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target-pass{test-lowering-configuration=true}))' -verify-diagnostics -split-input-file %s

#config = #iree_codegen.lowering.config<tile_sizes = [], native_vector_size = []>
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulSimt", workload_per_wg = [128, 32]>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) attributes {
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
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulSimt", workload_per_wg = [128, 32]>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) attributes {
      translation.info = #translation,
      workgroup_size = [48 : index, 8 : index, 1 : index]
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8x16xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<4x16xf32>
        // expected-error @+1 {{workgroup size is not 32 aligned for LLVMGPUMatmulSimt, got 48}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [], native_vector_size = []>
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulSimt", workload_per_wg = [128, 32]>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) attributes {
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
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulTensorCore", workload_per_wg = [32, 32]>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) attributes {
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
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulTensorCore", workload_per_wg = [32, 32]>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) attributes {
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
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulTensorCore", workload_per_wg = [32, 32]>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) attributes {
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
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulTensorCore", workload_per_wg = [32, 32]>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) attributes {
      translation.info = #translation,
      workgroup_size = [64 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<32x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x32xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<32x32xf32>
        // expected-error @+1 {{tensorcore size doesn't factor into second level tiling size for LLVMGPUMatmulTensorCore}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<32x16xf32>, memref<16x32xf32>)
          outs(%result: memref<32x32xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [[32, 32, 16]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulTensorCore", workload_per_wg = [32, 32]>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) attributes {
      translation.info = #translation,
      workgroup_size = [64 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<48x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x32xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<48x32xf32>
        // expected-error @+1 {{lhsShape doesn't factor into first level tiling size for LLVMGPUMatmulTensorCore}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<48x16xf32>, memref<16x32xf32>)
          outs(%result: memref<48x32xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [[32, 32, 16]], native_vector_size = []>
#translation = #iree_codegen.translation.info<"LLVMGPUMatmulTensorCore", workload_per_wg = [32, 32]>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @illegal layout(#executable_layout) attributes {
      translation.info = #translation,
      workgroup_size = [64 : index, 2 : index, 1 : index]
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<32x16xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<16x48xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<32x48xf32>
        // expected-error @+1 {{rhsShape doesn't factor into first level tiling size for LLVMGPUMatmulTensorCore}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<32x16xf32>, memref<16x48xf32>)
          outs(%result: memref<32x48xf32>)
        return
      }
    }
  }
}

// -----
