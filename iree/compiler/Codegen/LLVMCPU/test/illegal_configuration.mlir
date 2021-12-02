// RUN: iree-opt -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{test-lowering-configuration=true}))' -verify-diagnostics -split-input-file %s

#config = #iree_codegen.lowering.config<tile_sizes = [], native_vector_size = []>
#translation = #iree_codegen.translation.info<"CPUTensorToVectors", workload_per_wg = []>
hal.executable private @matmul_tensors  {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {}> {
    hal.executable.entry_point @illegal attributes {
      translation.info = #translation,
      interface = @io,
      ordinal = 0 : index
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan @io::@arg1[%c0] : memref<8x16xf32>
        %result = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4x16xf32>
        // expected-error @+1 {{expected 2 entries for workload_per_wg, but got 0}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [], native_vector_size = []>
#translation = #iree_codegen.translation.info<"CPUTensorToVectors", workload_per_wg = [1, 0]>
hal.executable private @matmul_tensors  {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {}> {
    hal.executable.entry_point @illegal attributes {
      translation.info = #translation,
      interface = @io,
      ordinal = 0 : index
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan @io::@arg1[%c0] : memref<8x16xf32>
        %result = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4x16xf32>
        // expected-error @+1 {{invalid to use 0 in workload_per_wg}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [], native_vector_size = []>
#translation = #iree_codegen.translation.info<"CPUTensorToVectors", workload_per_wg = [1, 1, 1, 1]>
hal.executable private @matmul_tensors  {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {}> {
    hal.executable.entry_point @illegal attributes {
      translation.info = #translation,
      interface = @io,
      ordinal = 0 : index
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan @io::@arg1[%c0] : memref<8x16xf32>
        %result = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4x16xf32>
        // expected-error @+1 {{workload_per_wg size should be less than 3}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [], native_vector_size = []>
#translation = #iree_codegen.translation.info<"CPUTensorToVectors", workload_per_wg = [1, 1]>
hal.executable private @matmul_tensors  {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {}> {
    hal.executable.entry_point @illegal attributes {
      translation.info = #translation,
      interface = @io,
      ordinal = 0 : index
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan @io::@arg1[%c0] : memref<8x16xf32>
        %result = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4x16xf32>
        // expected-error @+1 {{expected three levels of tile sizes for CPUTensorToVectors, got 0}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [[4, 8], [], []], native_vector_size = []>
#translation = #iree_codegen.translation.info<"CPUTensorToVectors", workload_per_wg = [8, 6]>
hal.executable private @matmul_tensors  {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {}> {
    hal.executable.entry_point @illegal attributes {
      translation.info = #translation,
      interface = @io,
      ordinal = 0 : index
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan @io::@arg1[%c0] : memref<8x16xf32>
        %result = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4x16xf32>
        // expected-error @+1 {{mismatch in distributed tile size value 4 at position 0 and workload_per_wg value 6}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering.config<tile_sizes = [[], [], [8, 8, 8]], native_vector_size = [4, 4, 4]>
#translation = #iree_codegen.translation.info<"CPUTensorToVectors", workload_per_wg = [8, 4]>
hal.executable private @matmul_tensors  {
  hal.interface @io {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding @arg1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer"
  }
  hal.executable.variant @llvm, target = #hal.executable.target<"llvm", "embedded-elf-x86_64", {}> {
    hal.executable.entry_point @illegal attributes {
      translation.info = #translation,
      interface = @io,
      ordinal = 0 : index
    }
    builtin.module {
      func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan @io::@arg0[%c0] : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan @io::@arg1[%c0] : memref<8x16xf32>
        %result = hal.interface.binding.subspan @io::@ret0[%c0] : memref<4x16xf32>
        // expected-error @+1 {{native_vector_size must be same as the last level of tiling}}
        linalg.matmul {lowering.config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}
