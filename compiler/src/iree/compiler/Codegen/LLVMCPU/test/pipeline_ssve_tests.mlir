// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-select-lowering-strategy, iree-llvmcpu-lower-executable-target)))' --split-input-file %s | FileCheck %s

// Check Armv9 Streaming SVE mode is enabled for the following pipelines:
//
//   * CPUBufferOpsTileAndVectorize
//   * CPUDoubleTilingPeelingExpert
//   * CPUConvTileAndDecomposeExpert
//   * CPUDoubleTilingExpert

#pipeline_layout = #hal.pipeline.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  cpu_features = "+sve,+sme",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-none-elf"
}>

hal.executable private @aarch64_ssve__cpu_buffer_ops_tile_and_vectorize {
  hal.executable.variant public @embedded_elf_arm_64 target(#executable_target_embedded_elf_arm_64_) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) attributes {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [1], [0], [0]]>,
      translation_info = #iree_codegen.translation_info<CPUBufferOpsTileAndVectorize>
    } {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      hal.return %arg1, %arg2, %arg2 : index, index, index
    }
    builtin.module {
      func.func @dispatch() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        %7 = tensor.empty() : tensor<1xf32>
        %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<1xf32>) -> tensor<1xf32>
        flow.dispatch.tensor.store %8, %6, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: @aarch64_ssve__cpu_buffer_ops_tile_and_vectorize
// CHECK: func.func @dispatch() attributes {arm_locally_streaming}

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  cpu_features = "+sve,+sme",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-none-elf"
}>

hal.executable private @aarch64_ssve__cpu_double_tiling_peeling_expert {
  hal.executable.variant public @embedded_elf_arm_64 target(#executable_target_embedded_elf_arm_64_) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) attributes {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [1], [0], [0]]>,
      translation_info = #iree_codegen.translation_info<CPUDoubleTilingPeelingExpert>
    } {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      hal.return %arg1, %arg2, %arg2 : index, index, index
    }
    builtin.module {
      func.func @dispatch() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        %7 = tensor.empty() : tensor<1xf32>
        %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<1xf32>) -> tensor<1xf32>
        flow.dispatch.tensor.store %8, %6, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: @aarch64_ssve__cpu_double_tiling_peeling_expert
// CHECK: func.func @dispatch() attributes {arm_locally_streaming}

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  cpu_features = "+sve,+sme",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-none-elf"
}>

hal.executable private @aarch64_ssve__cpu_double_tiling_expert {
  hal.executable.variant public @embedded_elf_arm_64 target(#executable_target_embedded_elf_arm_64_) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) attributes {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [1], [0], [0]]>,
      translation_info = #iree_codegen.translation_info<CPUDoubleTilingExpert>
    } {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      hal.return %arg1, %arg2, %arg2 : index, index, index
    }
    builtin.module {
      func.func @dispatch() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        %7 = tensor.empty() : tensor<1xf32>
        %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<1xf32>) -> tensor<1xf32>
        flow.dispatch.tensor.store %8, %6, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: @aarch64_ssve__cpu_double_tiling_expert
// CHECK: func.func @dispatch() attributes {arm_locally_streaming}

// -----

#pipeline_layout = #hal.pipeline.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  cpu_features = "+sve,+sme",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-none-elf"
}>

hal.executable private @aarch64_ssve__cpu_conv_tile_and_decompose_expert {
  hal.executable.variant public @embedded_elf_arm_64 target(#executable_target_embedded_elf_arm_64_) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) attributes {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [1], [0], [0]]>,
      translation_info = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
    } {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      hal.return %arg1, %arg2, %arg2 : index, index, index
    }
    builtin.module {
      func.func @dispatch() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        %7 = tensor.empty() : tensor<1xf32>
        %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<1xf32>) -> tensor<1xf32>
        flow.dispatch.tensor.store %8, %6, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: @aarch64_ssve__cpu_conv_tile_and_decompose_expert
// CHECK: func.func @dispatch() attributes {arm_locally_streaming}

// -----

// Check Armv9 Streaming SVE mode is not enabled if +sve is not
// specified.

#pipeline_layout = #hal.pipeline.layout<push_constants = 2, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

#executable_target_embedded_elf_arm_64_no_sve = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  cpu_features = "+sme",
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-none-elf"
}>

hal.executable private @aarch64_ssve_sve_disabled {
  hal.executable.variant public @embedded_elf_arm_64 target(#executable_target_embedded_elf_arm_64_no_sve) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) attributes {
      lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [1], [0], [0]]>,
      translation_info = #iree_codegen.translation_info<CPUBufferOpsTileAndVectorize>
    } {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      hal.return %arg1, %arg2, %arg2 : index, index, index
    }
    builtin.module {
      func.func @dispatch() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %cst_0 = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %6 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        %7 = tensor.empty() : tensor<1xf32>
        %8 = linalg.fill ins(%cst_0 : f32) outs(%7 : tensor<1xf32>) -> tensor<1xf32>
        flow.dispatch.tensor.store %8, %6, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1xf32>>
        return
      }
    }
  }
}

// CHECK-LABEL: @aarch64_ssve_sve_disabled
// CHECK-NOT: func.func @dispatch() attributes {arm_locally_streaming}
