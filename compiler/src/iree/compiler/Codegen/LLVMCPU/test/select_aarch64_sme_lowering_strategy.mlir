// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --iree-llvmcpu-enable-scalable-vectorization=true --split-input-file %s | FileCheck %s --check-prefixes=CHECK,WITH-SME
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --iree-llvmcpu-enable-scalable-vectorization=true --iree-llvmcpu-disable-arm-sme-tiling --split-input-file %s | FileCheck %s --check-prefixes=DISABLE-ARM-SME

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @transpose_f32(%2: tensor<32x32xf32>) -> tensor<32x32xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %3 = tensor.empty() : tensor<32x32xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<32x32xf32>) outs(%3 : tensor<32x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<32x32xf32>
  return %4 : tensor<32x32xf32>
}
//   CHECK: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [4, 32], vector_common_parallel = {{\[}}[4], [4]]>
//   CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
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
//   CHECK: #[[CONFIG:.+]] =  #iree_cpu.lowering_config<distribution = [4, 32], vector_common_parallel = {{\[}}[4], [4]]>
//   CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
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
//   CHECK: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [4, 32], vector_common_parallel = {{\[}}[2], [2]]>
//   CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
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
//   CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
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
//   CHECK: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
//       CHECK: func.func @transpose_unsupported_not_simple_transpose(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.generic
//  CHECK-SAME:     lowering_config = #[[CONFIG]]

// -----

#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @matmul_tensors(%7: tensor<?x?xf32>, %8: tensor<?x?xf32>, %9: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}
//  DISABLE-ARM-SME-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [8, [8], 0], vector_reduction = [0, 0, 4]>
//  DISABLE-ARM-SME-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
//      DISABLE-ARM-SME: func.func @matmul_tensors(
//  DISABLE-ARM-SME-SAME:     translation_info = #[[TRANSLATION]]
//       DISABLE-ARM-SME: linalg.matmul
//  DISABLE-ARM-SME-SAME:     lowering_config = #[[CONFIG]]

//   WITH-SME-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = {{\[}}[8], [8], 0], vector_reduction = [0, 0, 1]>
//   WITH-SME-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
//       WITH-SME: func.func @matmul_tensors(
//  WITH-SME-SAME:     translation_info = #[[TRANSLATION]]
//       WITH-SME: linalg.matmul
//  WITH-SME-SAME:     lowering_config = #[[CONFIG]]

// -----

// f64 matmul uses different SME tile sizes than f32: [4]x[8] instead of [8]x[8]
// (each SME tile only holds half as many f64 elements per row as f32 ones).
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @matmul_tensors_f64(%7: tensor<?x?xf64>, %8: tensor<?x?xf64>, %9: tensor<?x?xf64>) -> tensor<?x?xf64> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xf64>, tensor<?x?xf64>) outs(%9 : tensor<?x?xf64>) -> tensor<?x?xf64>
  return %10 : tensor<?x?xf64>
}
//  DISABLE-ARM-SME-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [8, [4], 0], vector_reduction = [0, 0, 2]>
//  DISABLE-ARM-SME-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
//      DISABLE-ARM-SME: func.func @matmul_tensors_f64(
//  DISABLE-ARM-SME-SAME:     translation_info = #[[TRANSLATION]]
//       DISABLE-ARM-SME: linalg.matmul
//  DISABLE-ARM-SME-SAME:     lowering_config = #[[CONFIG]]

//   WITH-SME-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = {{\[}}[4], [8], 0], vector_reduction = [0, 0, 1]>
//   WITH-SME-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
//       WITH-SME: func.func @matmul_tensors_f64(
//  WITH-SME-SAME:     translation_info = #[[TRANSLATION]]
//       WITH-SME: linalg.matmul
//  WITH-SME-SAME:     lowering_config = #[[CONFIG]]

// -----

// SME tiling only supports f32 and f64 matmuls, so an i8 matmul should fall
// back to the regular SVE tiling heuristic regardless of whether SME tiling
// is enabled or disabled.
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @matmul_tensors_i8i8_i32_with_sme(%7: tensor<?x?xi8>, %8: tensor<?x?xi8>, %9: tensor<?x?xi32>) -> tensor<?x?xi32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xi8>, tensor<?x?xi8>) outs(%9 : tensor<?x?xi32>) -> tensor<?x?xi32>
  return %10 : tensor<?x?xi32>
}
//  DISABLE-ARM-SME-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [8, [8], 0], vector_reduction = [0, 0, 4]>
//  DISABLE-ARM-SME-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
//      DISABLE-ARM-SME: func.func @matmul_tensors_i8i8_i32_with_sme(
//  DISABLE-ARM-SME-SAME:     translation_info = #[[TRANSLATION]]
//       DISABLE-ARM-SME: linalg.matmul
//  DISABLE-ARM-SME-SAME:     lowering_config = #[[CONFIG]]

//   WITH-SME-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [8, [8], 0], vector_reduction = [0, 0, 4]>
//   WITH-SME-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
//       WITH-SME: func.func @matmul_tensors_i8i8_i32_with_sme(
//  WITH-SME-SAME:     translation_info = #[[TRANSLATION]]
//       WITH-SME: linalg.matmul
//  WITH-SME-SAME:     lowering_config = #[[CONFIG]]

// -----

// A target with +sme but not +sve (i.e. no non-streaming SVE) still picks
// scalable SME tile sizes when SME tiling is enabled: matmul lowering does
// not itself require +sve. Note this goes through a different
// pre-processing strategy than the +sve,+sme case (Peeling, with an extra
// `cache_parallel` tiling level) because `hasAnySVEFeature` does not count
// +sme - see getVectorPreProcStrategy() in KernelDispatch.cpp.
//
// TODO: When SME tiling is disabled here there is no scalable
// fallback available (below), since the regular SVE tiling heuristic
// requires +sve. To target SSVE (streaming SVE) on a target that only has
// +sme, that fallback heuristic needs to learn to treat +sme as implying
// streaming-SVE support.
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @matmul_tensors_sme_no_sve(%7: tensor<?x?xf32>, %8: tensor<?x?xf32>, %9: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}
// SME tiling disabled: falls back to non-scalable NEON tile sizes, since the
// SVE fallback heuristic requires +sve (see TODO above).
//  DISABLE-ARM-SME-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [64, 64, 0], vector_common_parallel = [8, 8, 0], vector_reduction = [0, 0, 4]>
//  DISABLE-ARM-SME-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
//      DISABLE-ARM-SME: func.func @matmul_tensors_sme_no_sve(
//  DISABLE-ARM-SME-SAME:     translation_info = #[[TRANSLATION]]
//       DISABLE-ARM-SME: linalg.matmul
//  DISABLE-ARM-SME-SAME:     lowering_config = #[[CONFIG]]

// SME tiling enabled: still picks scalable [8]x[8] SME tile sizes.
//   WITH-SME-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [64, 64, 0], vector_common_parallel = {{\[}}[8], [8], 0], vector_reduction = [0, 0, 1]>
//   WITH-SME-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
//       WITH-SME: func.func @matmul_tensors_sme_no_sve(
//  WITH-SME-SAME:     translation_info = #[[TRANSLATION]]
//       WITH-SME: linalg.matmul
//  WITH-SME-SAME:     lowering_config = #[[CONFIG]]
