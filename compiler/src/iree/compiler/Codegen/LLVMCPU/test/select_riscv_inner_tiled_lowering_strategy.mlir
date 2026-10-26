// RUN: iree-opt --pass-pipeline='builtin.module(iree-llvmcpu-select-lowering-strategy)' --split-input-file %s | FileCheck %s

// Lowering strategy selection for `iree_codegen.inner_tiled` rooted dispatches
// on RISC-V V (RVV) targets. Distribution tile sizes are counts of inner
// tiles per workgroup, chosen by the cache-footprint cost model ported from
// `getMmt4dLoweringConfig` (see `selectInnerTiledDistribution`). Note the
// scalable-vectorization RVV strategy tests live in
// select_riscv_v_lowering_strategy.mlir; inner_tiled kinds bake a concrete
// `vlen` and do not use the scalable path.

#executable_target_riscv64 = #hal.executable.target<"llvm-cpu", "embedded-elf-riscv_64", {cpu_features = "+m,+a,+f,+d,+c,+v,+zvl256b", data_layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128", native_vector_size = 32 : index, target_triple = "riscv64-unknown-unknown-eabi-elf"}>

// A 540x516x332 f32 matmul after packing: 90 x 17 inner-tile iterations over
// K = 332. The 6x32 inner tile is MMA_RISCV_V_VFMACC_1x8VLsx1_F32_F32 with
// intrinsics_m = 6 at VLEN = 256. The model groups 18 inner tiles along M
// into one workgroup (18x6x332 LHS + 1x32x332 RHS + acc fit the 256KiB
// budget); N has 17 tiles, which is prime, so tilesN stays 1 and the dispatch
// gets 5 x 17 = 85 workgroups instead of one per inner tile (1530).
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
//   CHECK-DAG: #[[CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [18, 1, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//   CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Mmt4dTilingExpert>>
//       CHECK: func.func @inner_tiled_rvv_f32_540x516x332(
//  CHECK-SAME:     translation_info = #[[TRANSLATION]]
//       CHECK:   iree_codegen.inner_tiled
//  CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

// Dynamic outer extents: the cost model requires static M/N/K iteration
// bounds (the vector lowering chain cannot statically unroll dynamic ones),
// so distribution falls back to one inner tile per workgroup.
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
//   CHECK-DAG: #[[DYN_CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [1, 1, 0], vector_common_parallel = [1, 1, 0], vector_reduction = [0, 0, 1]>
//   CHECK-DAG: #[[DYN_TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_cpu.pipeline<Mmt4dTilingExpert>>
//       CHECK: func.func @inner_tiled_rvv_f32_dynamic(
//  CHECK-SAME:     translation_info = #[[DYN_TRANSLATION]]
//       CHECK:   iree_codegen.inner_tiled
//  CHECK-SAME:       lowering_config = #[[DYN_CONFIG]]
