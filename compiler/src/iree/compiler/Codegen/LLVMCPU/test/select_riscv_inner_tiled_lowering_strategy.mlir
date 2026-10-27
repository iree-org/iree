// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --iree-llvmcpu-number-of-threads=8 --split-input-file %s | FileCheck %s --check-prefix=T8
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --iree-llvmcpu-number-of-threads=4 --split-input-file %s | FileCheck %s --check-prefix=T4
// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --iree-llvmcpu-number-of-threads=1 --split-input-file %s | FileCheck %s --check-prefix=T1

// Lowering strategy selection for `iree_codegen.inner_tiled` rooted dispatches
// on RISC-V V (RVV), over the benchmark matrix from #24932: shapes
// 64x64x64, 256x256x256, 81x41x19 and 540x516x332 for a VLEN=256 f32 target
// (6x32 inner tile: MMA_RISCV_V_VFMACC_1x8VLsx1_F32_F32, intrinsics_m = 6),
// compiled at 8, 4 and 1 runtime threads. Distribution tile sizes are counts
// of inner tiles per workgroup, chosen by the cache-footprint cost model
// ported from `getMmt4dLoweringConfig` (see `selectInnerTiledDistribution`);
// its thread-underutilization term is what makes the small shapes sweep
// toward fewer, larger workgroup tiles as the thread count drops. The 1T
// benchmark flow additionally passes --iree-llvmcpu-disable-distribution,
// which zeroes these tiles downstream of this pass. Operand shapes are the
// packed forms emitted by data tiling (M/N padded up to inner-tile
// multiples: 64 -> 11x6, 81 -> 14x6, 41 -> 2x32, 516 -> 17x32).
// Note: the scalable-vectorization RVV strategy tests live in
// select_riscv_v_lowering_strategy.mlir; inner_tiled kinds bake a concrete
// `vlen` and do not use the scalable path.

#executable_target_riscv64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 32 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>

// 64x64x64: 11 x 2 inner-tile iterations over K = 64. At 8 threads the model
// keeps M at one tile per workgroup (11 workgroups); at 4 it consolidates M
// ([11, 1] -> N distributed only); at 1 it takes the whole matmul as one
// workgroup tile ([11, 2]).
func.func @inner_tiled_rvv_f32_64x64x64(%lhs: tensor<11x64x6x1x1xf32>, %rhs: tensor<2x64x32x1xf32>, %acc: tensor<11x2x6x1x32xf32>) -> tensor<11x2x6x1x32xf32> attributes {hal.executable.target = #executable_target_riscv64} {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) <
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1x8VLsx1_F32_F32, intrinsics_m = 6, vlen = 256>,
    semantics = #iree_cpu.mma_semantics<>
  > : tensor<11x64x6x1x1xf32>, tensor<2x64x32x1xf32> into tensor<11x2x6x1x32xf32>
  return %0 : tensor<11x2x6x1x32xf32>
}
//    T8-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 2, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//    T8-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Mmt4dTilingExpert>>
//        T8: func.func @inner_tiled_rvv_f32_64x64x64(
//   T8-SAME:     translation_info = #[[TRANSLATION]]
//        T8:   iree_codegen.inner_tiled
//   T8-SAME:       lowering_config = #[[CONFIG]]
//    T4-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [11, 1, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//        T4: func.func @inner_tiled_rvv_f32_64x64x64(
//        T4:   iree_codegen.inner_tiled
//   T4-SAME:       lowering_config = #[[CONFIG]]
//    T1-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [11, 2, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//        T1: func.func @inner_tiled_rvv_f32_64x64x64(
//        T1:   iree_codegen.inner_tiled
//   T1-SAME:       lowering_config = #[[CONFIG]]

// -----

// 256x256x256: 43 x 8 inner-tile iterations over K = 256. The whole-matrix
// undercount lets one workgroup take all 43 M tiles at every thread count;
// N (8 tiles) stays distributed.
#executable_target_riscv64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 32 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
func.func @inner_tiled_rvv_f32_256x256x256(%lhs: tensor<43x256x6x1x1xf32>, %rhs: tensor<8x256x32x1xf32>, %acc: tensor<43x8x6x1x32xf32>) -> tensor<43x8x6x1x32xf32> attributes {hal.executable.target = #executable_target_riscv64} {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) <
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1x8VLsx1_F32_F32, intrinsics_m = 6, vlen = 256>,
    semantics = #iree_cpu.mma_semantics<>
  > : tensor<43x256x6x1x1xf32>, tensor<8x256x32x1xf32> into tensor<43x8x6x1x32xf32>
  return %0 : tensor<43x8x6x1x32xf32>
}
//    T8-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [43, 1, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//        T8: func.func @inner_tiled_rvv_f32_256x256x256(
//        T8:   iree_codegen.inner_tiled
//   T8-SAME:       lowering_config = #[[CONFIG]]
//    T4-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [43, 1, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//        T4: func.func @inner_tiled_rvv_f32_256x256x256(
//        T4:   iree_codegen.inner_tiled
//   T4-SAME:       lowering_config = #[[CONFIG]]
//    T1-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [43, 1, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//        T1: func.func @inner_tiled_rvv_f32_256x256x256(
//        T1:   iree_codegen.inner_tiled
//   T1-SAME:       lowering_config = #[[CONFIG]]

// -----

// 81x41x19 (rect): 14 x 2 inner-tile iterations over K = 19. Small enough
// that the underutilization term dominates the whole sweep: 28 workgroups at
// 8 threads, 14 at 4, 1 at 1.
#executable_target_riscv64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 32 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
func.func @inner_tiled_rvv_f32_81x41x19(%lhs: tensor<14x19x6x1x1xf32>, %rhs: tensor<2x19x32x1xf32>, %acc: tensor<14x2x6x1x32xf32>) -> tensor<14x2x6x1x32xf32> attributes {hal.executable.target = #executable_target_riscv64} {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) <
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1x8VLsx1_F32_F32, intrinsics_m = 6, vlen = 256>,
    semantics = #iree_cpu.mma_semantics<>
  > : tensor<14x19x6x1x1xf32>, tensor<2x19x32x1xf32> into tensor<14x2x6x1x32xf32>
  return %0 : tensor<14x2x6x1x32xf32>
}
//    T8-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [2, 2, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//        T8: func.func @inner_tiled_rvv_f32_81x41x19(
//        T8:   iree_codegen.inner_tiled
//   T8-SAME:       lowering_config = #[[CONFIG]]
//    T4-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [7, 1, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//        T4: func.func @inner_tiled_rvv_f32_81x41x19(
//        T4:   iree_codegen.inner_tiled
//   T4-SAME:       lowering_config = #[[CONFIG]]
//    T1-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [14, 2, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//        T1: func.func @inner_tiled_rvv_f32_81x41x19(
//        T1:   iree_codegen.inner_tiled
//   T1-SAME:       lowering_config = #[[CONFIG]]

// -----

// 540x516x332 (large): 90 x 17 inner-tile iterations over K = 332 — the
// regression case from #24932/#25028 (previously one tile per workgroup:
// 1530 workgroups). N has 17 tiles, which is prime, so tilesN stays 1 and
// the budget-bound divisor search takes 18 M tiles: 5 x 17 = 85 workgroups
// at every thread count in the sweep.
#executable_target_riscv64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 32 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
func.func @inner_tiled_rvv_f32_540x516x332(%lhs: tensor<90x332x6x1x1xf32>, %rhs: tensor<17x332x32x1xf32>, %acc: tensor<90x17x6x1x32xf32>) -> tensor<90x17x6x1x32xf32> attributes {hal.executable.target = #executable_target_riscv64} {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) <
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1x8VLsx1_F32_F32, intrinsics_m = 6, vlen = 256>,
    semantics = #iree_cpu.mma_semantics<>
  > : tensor<90x332x6x1x1xf32>, tensor<17x332x32x1xf32> into tensor<90x17x6x1x32xf32>
  return %0 : tensor<90x17x6x1x32xf32>
}
//    T8-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [18, 1, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//        T8: func.func @inner_tiled_rvv_f32_540x516x332(
//        T8:   iree_codegen.inner_tiled
//   T8-SAME:       lowering_config = #[[CONFIG]]
//    T4-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [18, 1, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//        T4: func.func @inner_tiled_rvv_f32_540x516x332(
//        T4:   iree_codegen.inner_tiled
//   T4-SAME:       lowering_config = #[[CONFIG]]
//    T1-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [18, 1, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//        T1: func.func @inner_tiled_rvv_f32_540x516x332(
//        T1:   iree_codegen.inner_tiled
//   T1-SAME:       lowering_config = #[[CONFIG]]

// -----

// Dynamic outer extents: the cost model requires static M/N/K iteration
// bounds (the vector lowering chain cannot statically unroll dynamic ones),
// so distribution falls back to one inner tile per workgroup at every thread
// count.
#executable_target_riscv64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 32 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>
func.func @inner_tiled_rvv_f32_dynamic(%lhs: tensor<?x?x6x1x1xf32>, %rhs: tensor<?x?x32x1xf32>, %acc: tensor<?x?x6x1x32xf32>) -> tensor<?x?x6x1x32xf32> attributes {hal.executable.target = #executable_target_riscv64} {
  %0 = iree_codegen.inner_tiled ins(%lhs, %rhs) outs(%acc) <
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = [#linalg.iterator_type<parallel>,
                      #linalg.iterator_type<parallel>,
                      #linalg.iterator_type<reduction>],
    kind = #iree_cpu.data_tiled_mma_layout<intrinsic = MMA_RISCV_V_VFMACC_1x8VLsx1_F32_F32, intrinsics_m = 6, vlen = 256>,
    semantics = #iree_cpu.mma_semantics<>
  > : tensor<?x?x6x1x1xf32>, tensor<?x?x32x1xf32> into tensor<?x?x6x1x32xf32>
  return %0 : tensor<?x?x6x1x32xf32>
}
//    T8-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 1, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//        T8: func.func @inner_tiled_rvv_f32_dynamic(
//        T8:   iree_codegen.inner_tiled
//   T8-SAME:       lowering_config = #[[CONFIG]]
//    T4-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 1, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//        T4: func.func @inner_tiled_rvv_f32_dynamic(
//        T4:   iree_codegen.inner_tiled
//   T4-SAME:       lowering_config = #[[CONFIG]]
//    T1-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 1, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//        T1: func.func @inner_tiled_rvv_f32_dynamic(
//        T1:   iree_codegen.inner_tiled
//   T1-SAME:       lowering_config = #[[CONFIG]]
