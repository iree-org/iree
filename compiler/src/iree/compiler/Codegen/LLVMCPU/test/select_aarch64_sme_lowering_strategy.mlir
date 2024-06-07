// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --iree-llvmcpu-enable-scalable-vectorization=true --split-input-file %s | FileCheck %s

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @transpose_f32() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x32xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x32xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x32xf32>> -> tensor<32x32xf32>
  %3 = tensor.empty() : tensor<32x32xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<32x32xf32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : tensor<32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x32xf32>>
  return
}

//   CHECK: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4, 16], {{\[}}[4], [4]], [0, 0], [0, 0]]>
//   CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//       CHECK: func.func @transpose_f32()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.generic
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @transpose_f64() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x32xf64>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x32xf64>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x32xf64>> -> tensor<32x32xf64>
  %3 = tensor.empty() : tensor<32x32xf64>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<32x32xf64>) outs(%3 : tensor<32x32xf64>) {
  ^bb0(%in: f64, %out: f64):
    linalg.yield %in : f64
  } -> tensor<32x32xf64>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : tensor<32x32xf64> -> !flow.dispatch.tensor<writeonly:tensor<32x32xf64>>
  return
}

//   CHECK: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[4, 16], {{\[}}[2], [2]], [0, 0], [0, 0]]>
//   CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//       CHECK: func.func @transpose_f64()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.generic
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @transpose_unsupported_not_rank_2() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x4x8xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x8x4xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 4, 8], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4x8xf32>> -> tensor<2x4x8xf32>
  %3 = tensor.empty() : tensor<2x8x4xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<2x4x8xf32>) outs(%3 : tensor<2x8x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<2x8x4xf32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0], sizes = [2, 8, 4], strides = [1, 1, 1] : tensor<2x8x4xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x8x4xf32>>
  return
}

//   CHECK: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[2, 8, 4], [1, 4, 4], [0, 0, 0], [0, 0, 0]]>
//   CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//       CHECK: func.func @transpose_unsupported_not_rank_2
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.generic
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @transpose_unsupported_not_simple_transpose() attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<32x32xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<32x32xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<32x32xf32>> -> tensor<32x32xf32>
  %3 = tensor.empty() : tensor<32x32xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %out : f32
  } -> tensor<32x32xf32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : tensor<32x32xf32> -> !flow.dispatch.tensor<writeonly:tensor<32x32xf32>>
  return
}

//   CHECK: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[32, 32], {{\[}}4, 4], [0, 0], [0, 0]]>
//   CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<CPUDoubleTilingExpert>
//       CHECK: func.func @transpose_unsupported_not_simple_transpose()
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.generic
//  CHECK-SAME:     lowering_config = #[[CONFIG]]
