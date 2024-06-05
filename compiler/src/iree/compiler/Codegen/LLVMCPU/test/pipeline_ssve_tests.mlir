// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy, func.func(iree-llvmcpu-lower-executable-target))' --split-input-file %s | FileCheck %s

// Check Armv9 Streaming SVE mode is enabled for the following pipelines:
//
//   * CPUBufferOpsTileAndVectorize
//   * CPUConvTileAndDecomposeExpert
//   * CPUDoubleTilingExpert

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
module {
  func.func @dispatch() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_,
      translation_info = #iree_codegen.translation_info<CPUBufferOpsTileAndVectorize>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1xf32>>
    %2 = tensor.empty() : tensor<1xf32>
    %3 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [1], [0], [0]]>}
        ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
    flow.dispatch.tensor.store %3, %1, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1xf32>>
    return
  }
}

// CHECK: func.func @dispatch()
// CHECK-SAME: arm_locally_streaming

// -----
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
module {
  func.func @dispatch() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_,
      translation_info = #iree_codegen.translation_info<CPUDoubleTilingExpert>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1xf32>>
    %2 = tensor.empty() : tensor<1xf32>
    %3 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [1], [0], [0]]>}
        ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
    flow.dispatch.tensor.store %3, %1, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1xf32>>
    return
  }
}

// CHECK: func.func @dispatch()
// CHECK-SAME: arm_locally_streaming

// -----
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
module {
  func.func @dispatch() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_,
      translation_info = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1xf32>>
    %2 = tensor.empty() : tensor<1xf32>
    %3 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [1], [0], [0]]>}
        ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
    flow.dispatch.tensor.store %3, %1, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1xf32>>
    return
  }
}

// CHECK: func.func @dispatch()
// CHECK-SAME: arm_locally_streaming

// -----
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
module {
  func.func @dispatch() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_,
      translation_info = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.constant.load[0] : i32
    %1 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<1xf32>>
    %2 = tensor.empty() : tensor<1xf32>
    %3 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0], [1], [0], [0]]>}
        ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
    flow.dispatch.tensor.store %3, %1, offsets = [0], sizes = [1], strides = [1] : tensor<1xf32> -> !flow.dispatch.tensor<readwrite:tensor<1xf32>>
    return
  }
}

// CHECK: func.func @dispatch()
// CHECK-NOT: arm_locally_streaming
