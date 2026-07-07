// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --iree-llvmcpu-enable-scalable-vectorization=true --split-input-file %s | FileCheck %s --check-prefixes=CHECK,WITH-SME
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --iree-llvmcpu-enable-scalable-vectorization=true --iree-llvmcpu-disable-arm-sme-tiling --split-input-file %s | FileCheck %s --check-prefixes=CHECK,DISABLE-ARM-SME

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
// streaming-SVE support. Note --iree-llvmcpu-force-arm-streaming does not
// change this: it's consumed by a later pass in the full lowering-to-LLVM
// pipeline (addLowerToLLVMPasses), not by iree-llvmcpu-select-lowering-
// strategy, so it has no effect on the tile sizes selected here.
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @matmul_tensors_sme_no_sve(%7: tensor<?x?xf32>, %8: tensor<?x?xf32>, %9: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}
// SME tiling disabled: falls back to non-scalable NEON tile sizes, since the
// SVE fallback heuristic requires +sve (see TODO above).
//  DISABLE-ARM-SME-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [64, 64, 0], vector_common_parallel = [8, 8, 0], vector_reduction = [0, 0, 4]>
// SME tiling enabled: still picks scalable [8]x[8] SME tile sizes.
//   WITH-SME-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<cache_parallel = [64, 64, 0], distribution = [64, 64, 0], vector_common_parallel = {{\[}}[8], [8], 0], vector_reduction = [0, 0, 1]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>
//       CHECK: func.func @matmul_tensors_sme_no_sve(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]
