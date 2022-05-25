// RUN: iree-opt --pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{test-lowering-configuration=true}))' --verify-diagnostics --split-input-file %s

#config = #iree_codegen.lowering_config<tile_sizes = []>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {}> {
    hal.executable.entry_point @illegal layout(#executable_layout)  {
      translation_info = #translation
    }
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8x16xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<4x16xf32>
        // expected-error @+1 {{expected three tiling sizes, got 0}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[4, 8], [8, 8, 0], [0, 0, 8]], tile_interchange = [[], [], []], native_vector_size = [0, 0, 4]>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {}> {
    hal.executable.entry_point @illegal layout(#executable_layout)  {
      translation_info = #translation
    }
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8x16xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<4x16xf32>
        // expected-error @+1 {{native_vector_size must be empty}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [8, 32, 16], [0, 0, 16]]>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {}> {
    hal.executable.entry_point @illegal layout(#executable_layout)  {
      translation_info = #translation
    }
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8x16xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<4x16xf32>
        // expected-error @+1 {{expected only parallel dims to be set in the second tiling sizes, got 2-th tile size set}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [8, 0, 0], [0, 16, 16]]>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {}> {
    hal.executable.entry_point @illegal layout(#executable_layout)  {
      translation_info = #translation
    }
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8x16xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<4x16xf32>
        // expected-error @+1 {{only reduction dims to be set in the third tiling sizes, got 1-th tile size set}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[4, 8], [8, 8, 0], [0, 0, 8]], tile_interchange = [[1], [], []]>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {}> {
    hal.executable.entry_point @illegal layout(#executable_layout)  {
      translation_info = #translation
    }
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8x16xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<4x16xf32>
        // expected-error @+1 {{expected [0, 2) to be set exactly once in interchange #0}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

// The constraints of CPUDoubleTilingPadExpert is as same as
// CPUDoubleTilingExpert, checking one test it enough.
#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [8, 32, 16], [0, 0, 16]]>
#translation = #iree_codegen.translation_info<CPUDoubleTilingPadExpert>
#executable_layout = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {}> {
    hal.executable.entry_point @illegal layout(#executable_layout)  {
      translation_info = #translation
    }
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8x16xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<4x16xf32>
        // expected-error @+1 {{expected only parallel dims to be set in the second tiling sizes, got 2-th tile size set}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}
