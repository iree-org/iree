// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" %s | FileCheck %s

// ===----------------------------------------------------------------------===//
// Shared Attributes
// ===----------------------------------------------------------------------===//

// Note the ukernel provider being specified in the executable target. This should be used to determine the data tiling.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {
  abi = "hip",
  iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>,
  iree_codegen.target_info = #iree_gpu.target<
    arch = "gfx942",
    features = "",
    wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8,
    storage =  b64|b32|b16|b8,
    subgroup =  shuffle|arithmetic,
    dot =  dp4xi8toi32,
    mma = [<MFMA_F32_16x16x16_BF16>, <MFMA_F32_32x32x8_BF16>, <MFMA_F32_16x16x32_F8E5M2FNUZ>,
           <MFMA_F32_16x16x32_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_16x16x32_F8E4M3FNUZ>,
           <MFMA_F32_16x16x32_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_F32_32x32x16_F8E5M2FNUZ>,
           <MFMA_F32_32x32x16_F8E5M2FNUZ_F8E4M3FNUZ>, <MFMA_F32_32x32x16_F8E4M3FNUZ>,
           <MFMA_F32_32x32x16_F8E4M3FNUZ_F8E5M2FNUZ>, <MFMA_I32_16x16x32_I8>,
           <MFMA_I32_32x32x16_I8>, <MFMA_F64_16x16x4_F64>, <MFMA_F32_16x16x4_F32>,
           <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>
          ],
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

#encoding_lhs_f16 = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs_f16 = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result_f16 = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>

func.func @matmul_f16_f16_f32_large_lowering_ukernel_provider(%arg0: tensor<?x?xf16, #encoding_lhs_f16>, %arg1: tensor<?x?xf16, #encoding_rhs_f16>, %arg2: tensor<?x?xf32, #encoding_result_f16>) -> tensor<?x?xf32, #encoding_result_f16> attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xf16, #encoding_lhs_f16>, tensor<?x?xf16, #encoding_rhs_f16>)
      outs(%arg2 : tensor<?x?xf32, #encoding_result_f16>)
      -> tensor<?x?xf32, #encoding_result_f16>
  return %0 : tensor<?x?xf32, #encoding_result_f16>
}
// CHECK-LABEL: matmul_f16_f16_f32_large_lowering_ukernel_provider
// CHECK:      iree_codegen.inner_tiled
// CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:     kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x16_F16, intrinsics_m = 8, subgroups_m = 2, intrinsics_n = 4, subgroups_n = 4>

#encoding_lhs_f8_medium = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs_f8_medium = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result_f8_medium = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>

func.func @matmul_f8_f8_f32_medium_lowering_ukernel_provider(%arg0: tensor<?x?xf8E4M3FNUZ, #encoding_lhs_f8_medium>, %arg1: tensor<?x?xf8E4M3FNUZ, #encoding_rhs_f8_medium>, %arg2: tensor<?x?xf32, #encoding_result_f8_medium>) -> tensor<?x?xf32, #encoding_result_f8_medium> attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xf8E4M3FNUZ, #encoding_lhs_f8_medium>, tensor<?x?xf8E4M3FNUZ, #encoding_rhs_f8_medium>)
      outs(%arg2 : tensor<?x?xf32, #encoding_result_f8_medium>)
      -> tensor<?x?xf32, #encoding_result_f8_medium>
  return %0 : tensor<?x?xf32, #encoding_result_f8_medium>
}
// CHECK-LABEL: matmul_f8_f8_f32_medium_lowering_ukernel_provider
// CHECK:      iree_codegen.inner_tiled
// CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:     kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 8, intrinsics_k = 2, operands_interleaving_intrinsics_k = [0, 1]>

#encoding_lhs_f8_large = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, 2048, ?]>
#encoding_rhs_f8_large = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, 2048, ?]>
#encoding_result_f8_large = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, 2048, ?]>

func.func @matmul_f8_f8_f32_large_lowering_ukernel_provider(%arg0: tensor<?x?xf8E4M3FNUZ, #encoding_lhs_f8_large>, %arg1: tensor<?x2048xf8E4M3FNUZ, #encoding_rhs_f8_large>, %arg2: tensor<?x2048xf32, #encoding_result_f8_large>) -> tensor<?x2048xf32, #encoding_result_f8_large> attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xf8E4M3FNUZ, #encoding_lhs_f8_large>, tensor<?x2048xf8E4M3FNUZ, #encoding_rhs_f8_large>)
      outs(%arg2 : tensor<?x2048xf32, #encoding_result_f8_large>)
      -> tensor<?x2048xf32, #encoding_result_f8_large>
  return %0 : tensor<?x2048xf32, #encoding_result_f8_large>
}
// CHECK-LABEL: matmul_f8_f8_f32_large_lowering_ukernel_provider
// CHECK:      iree_codegen.inner_tiled
// CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:     kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ,  intrinsics_m = 8, subgroups_m = 2, intrinsics_n = 4, subgroups_n = 4, operands_interleaving_intrinsics_k = [0, 1]>

// Test that with M=4, no ukernel matches due to the M dimension constraint,
// so it falls back to the default heuristic which should produce much smaller
// tile sizes to avoid excessive padding.

#encoding_lhs_f8_small_m = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [4, 4096, 4096]>
#encoding_rhs_f8_small_m = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [4, 4096, 4096]>
#encoding_result_f8_small_m = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [4, 4096, 4096]>

func.func @matmul_f8_f8_f32_small_m_no_ukernel_match(%arg0: tensor<4x4096xf8E4M3FNUZ, #encoding_lhs_f8_small_m>, %arg1: tensor<4096x4096xf8E4M3FNUZ, #encoding_rhs_f8_small_m>, %arg2: tensor<4x4096xf32, #encoding_result_f8_small_m>) -> tensor<4x4096xf32, #encoding_result_f8_small_m> attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<4x4096xf8E4M3FNUZ, #encoding_lhs_f8_small_m>, tensor<4096x4096xf8E4M3FNUZ, #encoding_rhs_f8_small_m>)
      outs(%arg2 : tensor<4x4096xf32, #encoding_result_f8_small_m>)
      -> tensor<4x4096xf32, #encoding_result_f8_small_m>
  return %0 : tensor<4x4096xf32, #encoding_result_f8_small_m>
}
// CHECK-LABEL: matmul_f8_f8_f32_small_m_no_ukernel_match
// CHECK:      iree_codegen.inner_tiled
// CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-NOT:     intrinsics_m = 8, subgroups_m = 2
// CHECK-NOT:     intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 8

// Test that with M=32 (just at the boundary), the medium ukernel matches.

#encoding_lhs_f8_m32 = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [32, 1024, 4096]>
#encoding_rhs_f8_m32 = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [32, 1024, 4096]>
#encoding_result_f8_m32 = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [32, 1024, 4096]>

func.func @matmul_f8_f8_f32_m32_medium_ukernel_match(%arg0: tensor<32x4096xf8E4M3FNUZ, #encoding_lhs_f8_m32>, %arg1: tensor<4096x1024xf8E4M3FNUZ, #encoding_rhs_f8_m32>, %arg2: tensor<32x1024xf32, #encoding_result_f8_m32>) -> tensor<32x1024xf32, #encoding_result_f8_m32> attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<32x4096xf8E4M3FNUZ, #encoding_lhs_f8_m32>, tensor<4096x1024xf8E4M3FNUZ, #encoding_rhs_f8_m32>)
      outs(%arg2 : tensor<32x1024xf32, #encoding_result_f8_m32>)
      -> tensor<32x1024xf32, #encoding_result_f8_m32>
  return %0 : tensor<32x1024xf32, #encoding_result_f8_m32>
}
// CHECK-LABEL: matmul_f8_f8_f32_m32_medium_ukernel_match
// CHECK:      iree_codegen.inner_tiled
// CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:     kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 8, intrinsics_k = 2, operands_interleaving_intrinsics_k = [0, 1]>
