// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --iree-llvmcpu-enable-scalable-vectorization=true --split-input-file %s | FileCheck %s

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @transpose_f32(%2: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %3 = tensor.empty() : tensor<32x32xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<32x32xf32>
  return %4 : tensor<32x32xf32>
}
//   CHECK: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [4, 16], vector_common_parallel = {{\[}}[4], [4]]>
//   CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//       CHECK: func.func @transpose_f32(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.generic
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @transpose_output_indexing_map_f32(%2: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %3 = tensor.empty() : tensor<32x32xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1, d0)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<32x32xf32>
  return %4 : tensor<32x32xf32>
}
//   CHECK: #[[CONFIG:.+]] =  #iree_cpu.lowering_config<distribution = [4, 16], vector_common_parallel = {{\[}}[4], [4]]>
//   CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//       CHECK: func.func @transpose_output_indexing_map_f32(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.generic
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @transpose_f64(%2: tensor<32x32xf64>) -> tensor<32x32xf64> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %3 = tensor.empty() : tensor<32x32xf64>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<32x32xf64>) outs(%3 : tensor<32x32xf64>) {
  ^bb0(%in: f64, %out: f64):
    linalg.yield %in : f64
   } -> tensor<32x32xf64>
   return %4 : tensor<32x32xf64>
}
//   CHECK: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [4, 16], vector_common_parallel = {{\[}}[2], [2]]>
//   CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//       CHECK: func.func @transpose_f64(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.generic
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @transpose_unsupported_not_rank_2(%2: tensor<2x4x8xf32>) -> tensor<2x8x4xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %3 = tensor.empty() : tensor<2x8x4xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<2x4x8xf32>) outs(%3 : tensor<2x8x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
   } -> tensor<2x8x4xf32>
   return %4 : tensor<2x8x4xf32>
}
//   CHECK: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [2, 8, 4], vector_common_parallel = [1, 4, 4]>
//   CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//       CHECK: func.func @transpose_unsupported_not_rank_2
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.generic
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @transpose_unsupported_not_simple_transpose(%2: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %3 = tensor.empty() : tensor<32x32xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %out : f32
   } -> tensor<32x32xf32>
   return %4 : tensor<32x32xf32>
}
//   CHECK: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [32, 32], vector_common_parallel = [4, 4]>
//   CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
//       CHECK: func.func @transpose_unsupported_not_simple_transpose(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.generic
//  CHECK-SAME:     lowering_config = #[[CONFIG]]
