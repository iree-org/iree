// Row 1: +sve=T, scalable-vec=T, force-arm-streaming=T, disable-sme-tiling=F -> SSVE
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true \
// RUN:   --iree-llvmcpu-force-arm-streaming=true \
// RUN:   --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SVE-FN,VECTOR

// Row 2: +sve=T, scalable-vec=T, force-arm-streaming=T, disable-sme-tiling=T -> SSVE
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true \
// RUN:   --iree-llvmcpu-force-arm-streaming=true \
// RUN:   --iree-llvmcpu-disable-arm-sme-tiling=true \
// RUN:   --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SVE-FN,VECTOR

// Row 3: +sve=T, scalable-vec=T, force-arm-streaming=F, disable-sme-tiling=F -> SVE
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true \
// RUN:   --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SVE-FN,VECTOR

// Row 4: +sve=T, scalable-vec=T, force-streaming=F, disable-sme-tiling=T -> SVE
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true \
// RUN:   --iree-llvmcpu-disable-arm-sme-tiling=true \
// RUN:   --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SVE-FN,VECTOR

// Row 5: +sve=T, scalable-vec=F, disable-sme-tiling=T, force-streaming=T -> NEON
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-disable-arm-sme-tiling=true \
// RUN:   --iree-llvmcpu-force-arm-streaming=true \
// RUN:   --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SVE-FN,SCALAR
// Row 6: +sve=T, scalable-vec=F, disable-sme-tiling=T, force-streaming=F -> NEON
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-disable-arm-sme-tiling=true \
// RUN:   --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SVE-FN,SCALAR
// Row 7: +sve=T, scalable-vec=F, disable-sme-tiling=F, force-streaming=T -> NEON
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-force-arm-streaming=true \
// RUN:   --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SVE-FN,SCALAR
// Row 8: +sve=T, scalable-vec=F, disable-sme-tiling=F, force-streaming=F -> NEON
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SVE-FN,SCALAR

// ===========================================================================
// +sve=F group: tested against @elementwise_add_sme_only
// ===========================================================================

// Row 9: +sve=F, scalable-vec=T, force-streaming=T, disable-sme-tiling=T -> VECTOR
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true \
// RUN:   --iree-llvmcpu-force-arm-streaming=true \
// RUN:   --iree-llvmcpu-disable-arm-sme-tiling=true \
// RUN:   --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SME-ONLY-FN,VECTOR
// Row 10: +sve=F, scalable-vec=T, force-streaming=T, disable-sme-tiling=F -> VECTOR
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true \
// RUN:   --iree-llvmcpu-force-arm-streaming=true \
// RUN:   --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SME-ONLY-FN,VECTOR
// Row 11: +sve=F, scalable-vec=T, force-streaming=F, disable-sme-tiling=T -> NEON
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true \
// RUN:   --iree-llvmcpu-disable-arm-sme-tiling=true \
// RUN:   --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SME-ONLY-FN,SCALAR
// Row 12: +sve=F, scalable-vec=F, force-streaming=T, disable-sme-tiling=T -> NEON
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-force-arm-streaming=true \
// RUN:   --iree-llvmcpu-disable-arm-sme-tiling=true \
// RUN:   --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SME-ONLY-FN,SCALAR
// Row 13: +sve=F, scalable-vec=T, force-streaming=F, disable-sme-tiling=F -> NEON
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-enable-scalable-vectorization=true \
// RUN:   --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SME-ONLY-FN,SCALAR
// Row 14: +sve=F, scalable-vec=F, force-streaming=F, disable-sme-tiling=T -> NEON
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-disable-arm-sme-tiling=true \
// RUN:   --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SME-ONLY-FN,SCALAR
// Row 15: +sve=F, scalable-vec=F, force-streaming=T, disable-sme-tiling=F -> NEON
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --iree-llvmcpu-force-arm-streaming=true \
// RUN:   --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SME-ONLY-FN,SCALAR
// Row 16: +sve=F, scalable-vec=F, force-streaming=F, disable-sme-tiling=F -> NEON
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefixes=CHECK,SME-ONLY-FN,SCALAR

#executable_target_sve_sme = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sve,+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>

func.func @elementwise_add_sve_sme(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32>
    attributes {hal.executable.target = #executable_target_sve_sme} {
  %init = tensor.empty() : tensor<1024xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<1024xf32>, tensor<1024xf32>)
    outs(%init : tensor<1024xf32>) {
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %0 = arith.addf %in1, %in2 : f32
    linalg.yield %0 : f32
  } -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// -----

#executable_target_sme_only = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {cpu_features = "+sme", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "aarch64-none-elf"}>

func.func @elementwise_add_sme_only(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32>
    attributes {hal.executable.target = #executable_target_sme_only} {
  %init = tensor.empty() : tensor<1024xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%arg0, %arg1 : tensor<1024xf32>, tensor<1024xf32>)
    outs(%init : tensor<1024xf32>) {
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %0 = arith.addf %in1, %in2 : f32
    linalg.yield %0 : f32
  } -> tensor<1024xf32>
  return %result : tensor<1024xf32>
}

// ===========================================================================
// CHECK lines
// ===========================================================================

// VECTOR-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<{{.*}}vector_common_parallel = {{\[}}[4]]{{.*}}>
// SCALAR-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<{{.*}}vector_common_parallel = [4]{{.*}}>
// VECTOR-DAG:  #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>>
// SCALAR-DAG:  #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<DoubleTilingExpert>, {enable_loop_peeling}>

// SVE-FN:       func.func @elementwise_add_sve_sme(
// SVE-FN-SAME:      translation_info = #[[TRANSLATION]]
// SME-ONLY-FN:       func.func @elementwise_add_sme_only(
// SME-ONLY-FN-SAME:      translation_info = #[[TRANSLATION]]

// CHECK:       linalg.generic
// CHECK-SAME:      lowering_config = #[[CONFIG]]
