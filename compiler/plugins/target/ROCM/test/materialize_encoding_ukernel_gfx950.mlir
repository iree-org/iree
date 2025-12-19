// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" %s | FileCheck %s

// ===----------------------------------------------------------------------===//
// Shared Attributes
// ===----------------------------------------------------------------------===//

// Note the ukernel provider being specified in the executable target. This should be used to determine the data tiling.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {
  abi = "hip",
  iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>,
  iree_codegen.target_info = #iree_gpu.target<
    arch = "gfx950",
    features = "",
    wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8,
    storage =  b64|b32|b16|b8,
    subgroup =  shuffle|arithmetic,
    dot =  dp4xi8toi32,
    mma = [<MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>],
    subgroup_size_choices = [64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647],
    max_load_instruction_bits = 128,
    simds_per_wgp = 4,
    vgpr_space_bits = 16384>
  >,
  iree_codegen.ukernel_provider = #rocm.tensor_ukernel_provider,
  ukernels = "none"
}>

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// ===----------------------------------------------------------------------===//
// Tests
// ===----------------------------------------------------------------------===//

// Test that the f16 dt matmul ukernel (which has archs = ["gfx942"]) is NOT
// matched on gfx950, demonstrating the archs filtering works correctly.
// The ukernel has: intrinsics_m = 8, subgroups_m = 2, intrinsics_n = 4, subgroups_n = 4
// gfx950 should fall back to default heuristic with different tile sizes.

#encoding_lhs_f16 = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs_f16 = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result_f16 = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>

func.func @matmul_f16_f16_f32_gfx950_no_ukernel_match(%arg0: tensor<?x?xf16, #encoding_lhs_f16>, %arg1: tensor<?x?xf16, #encoding_rhs_f16>, %arg2: tensor<?x?xf32, #encoding_result_f16>) -> tensor<?x?xf32, #encoding_result_f16> attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xf16, #encoding_lhs_f16>, tensor<?x?xf16, #encoding_rhs_f16>)
      outs(%arg2 : tensor<?x?xf32, #encoding_result_f16>)
      -> tensor<?x?xf32, #encoding_result_f16>
  return %0 : tensor<?x?xf32, #encoding_result_f16>
}
// CHECK-LABEL: matmul_f16_f16_f32_gfx950_no_ukernel_match
// CHECK:      iree_codegen.inner_tiled
// CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-NOT:      kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x16_F16, intrinsics_m = 8, subgroups_m = 2, intrinsics_n = 4, subgroups_n = 4>
