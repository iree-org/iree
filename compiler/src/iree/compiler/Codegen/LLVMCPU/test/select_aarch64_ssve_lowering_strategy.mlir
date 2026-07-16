// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --iree-llvmcpu-enable-scalable-vectorization=true --split-input-file %s | FileCheck %s --check-prefixes=CHECK,WITH-SME
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --iree-llvmcpu-enable-scalable-vectorization=true --iree-llvmcpu-disable-arm-sme-tiling --split-input-file %s | FileCheck %s --check-prefixes=CHECK,DISABLE-ARM-SME

// A target with +sme but not +sve (i.e. no non-streaming SVE) picks scalable
// tile sizes just like the +sve,+sme case, whether SME tiling is enabled or
// disabled: SME implies streaming SVE (SSVE), which supports scalable
// vectorization and masking on its own, without requiring +sve - see
// getMatmulVectorSizes() and getVectorPreProcStrategy() in KernelDispatch.cpp.
// Streaming mode itself is forced unconditionally for such targets in
// addLowerToLLVMPasses() (Passes.cpp), since they have no non-streaming SVE
// to fall back on.
#executable_target_embedded_elf_arm_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>
func.func @matmul_tensors_sme_no_sve(%7: tensor<?x?xf32>, %8: tensor<?x?xf32>, %9: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {hal.executable.target = #executable_target_embedded_elf_arm_64_} {
  %10 = linalg.matmul ins(%7, %8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%9 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %10 : tensor<?x?xf32>
}
// SME tiling disabled: falls back to the regular SVE-style tiling heuristic,
// scalable on the N dimension only (same as the +sve,+sme case).
//  DISABLE-ARM-SME-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = [8, [8], 0], vector_reduction = [0, 0, 4]>
// SME tiling enabled: picks scalable [8]x[8] SME tile sizes.
//   WITH-SME-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [64, 64, 0], vector_common_parallel = {{\[}}[8], [8], 0], vector_reduction = [0, 0, 1]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
//       CHECK: func.func @matmul_tensors_sme_no_sve(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK: linalg.matmul
//  CHECK-SAME:     lowering_config = #[[CONFIG]]
