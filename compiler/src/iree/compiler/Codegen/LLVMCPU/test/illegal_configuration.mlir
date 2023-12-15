// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-select-lowering-strategy)))' --verify-diagnostics --split-input-file %s

#config = #iree_codegen.lowering_config<tile_sizes = []>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @llvm target(#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>) {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {translation_info = #translation}
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8x16xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<4x16xf32>
        // expected-error @+1 {{expected four tiling levels, got 0}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[4, 8], [8, 8, 0], [0, 0, 8], [0, 0, 0]], native_vector_size = [0, 0, 4]>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @llvm target(#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>) {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {translation_info = #translation}
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

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [8, 32, 16], [0, 0, 16], [0, 0, 0]]>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @llvm target(#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>) {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {translation_info = #translation}
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8x16xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<4x16xf32>
        // expected-error @+1 {{expected only parallel dims to be set in the second tiling level, got 2-th tile size set}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [8, 0, 0], [0, 16, 16], [0, 0, 0]]>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @llvm target(#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>) {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {translation_info = #translation}
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<4x8xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<8x16xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<4x16xf32>
        // expected-error @+1 {{only reduction dims to be set in the third tiling level, got 1-th tile size set}}
        linalg.matmul {lowering_config = #config} ins(%lhs, %rhs : memref<4x8xf32>, memref<8x16xf32>)
          outs(%result: memref<4x16xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [{sizes=[4, 8], interchange=[1]}, [8, 8, 0], [0, 0, 8], [0, 0, 0]]>
#translation = #iree_codegen.translation_info<CPUDoubleTilingExpert>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @matmul_tensors {
  hal.executable.variant @llvm target(#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>) {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {translation_info = #translation}
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

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 7, 7, 64, 0, 0, 0], [6, 1, 7, 32, 0, 0, 0], [0, 0, 0, 0, 3, 3, 4], [0, 0, 0, 0, 0, 0, 0]]>
#translation = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @conv_2d_nhwc_hwcf {
  hal.executable.variant @llvm target(#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>) {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {translation_info = #translation}
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<36x9x9x512xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<3x3x512x512xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<36x7x7x512xf32>
        // expected-error @+1 {{can't decompose the conv op}}
        linalg.conv_2d_nhwc_hwcf {lowering_config = #config}
          ins(%lhs, %rhs : memref<36x9x9x512xf32>, memref<3x3x512x512xf32>)
          outs(%result: memref<36x7x7x512xf32>)
        return
      }
    }
  }
}

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 7, 64, 0, 0], [1, 1, 7, 8, 0, 0], [0, 0, 0, 0, 5, 5], [0, 0, 0, 0, 0, 0]]>
#translation = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @depthwise_conv_2d_nhwc_hwc {
  hal.executable.variant @llvm target(#hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {}>) {
    hal.executable.export @illegal layout(#pipeline_layout) attributes {translation_info = #translation}
    builtin.module {
      func.func @illegal() {
        %c0 = arith.constant 0 : index
        %lhs = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) : memref<1x11x11x576xf32>
        %rhs = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) : memref<5x5x576xf32>
        %result = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) : memref<1x7x7x576xf32>
        // expected-error @+1 {{can't decompose the conv op}}
        linalg.depthwise_conv_2d_nhwc_hwc {lowering_config = #config, dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
          ins(%lhs, %rhs : memref<1x11x11x576xf32>, memref<5x5x576xf32>)
          outs(%result: memref<1x7x7x576xf32>)
        return
      }
    }
  }
}
