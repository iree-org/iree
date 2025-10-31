// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --split-input-file %s | FileCheck %s

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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#pipeline_layout_3 = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @matmul_f16_f16_f32_large_lowering_ukernel_provider(%arg0: tensor<?x?xf16, #encoding_lhs>, %arg1: tensor<?x?xf16, #encoding_rhs>, %arg2: tensor<?x?xf32, #encoding_result>) -> tensor<?x?xf32, #encoding_result> attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xf16, #encoding_lhs>, tensor<?x?xf16, #encoding_rhs>)
      outs(%arg2 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  return %0 : tensor<?x?xf32, #encoding_result>
}
// CHECK-LABEL: matmul_f16_f16_f32_large_lowering_ukernel_provider
// CHECK:      iree_codegen.inner_tiled
// CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:     kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x16_F16, intrinsics_m = 8, subgroups_m = 2, intrinsics_n = 4, subgroups_n = 4>

// -----

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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#pipeline_layout_3 = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @matmul_f8_f8_f32_medium_lowering_ukernel_provider(%arg0: tensor<?x?xf8E4M3FNUZ, #encoding_lhs>, %arg1: tensor<?x?xf8E4M3FNUZ, #encoding_rhs>, %arg2: tensor<?x?xf32, #encoding_result>) -> tensor<?x?xf32, #encoding_result> attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xf8E4M3FNUZ, #encoding_lhs>, tensor<?x?xf8E4M3FNUZ, #encoding_rhs>)
      outs(%arg2 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  return %0 : tensor<?x?xf32, #encoding_result>
}
// CHECK-LABEL: matmul_f8_f8_f32_medium_lowering_ukernel_provider
// CHECK:      iree_codegen.inner_tiled
// CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:     kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 8, intrinsics_k = 2>

// -----

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
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, 2048, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, 2048, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, 2048, ?]>
#pipeline_layout_3 = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @matmul_f8_f8_f32_large_lowering_ukernel_provider(%arg0: tensor<?x?xf8E4M3FNUZ, #encoding_lhs>, %arg1: tensor<?x2048xf8E4M3FNUZ, #encoding_rhs>, %arg2: tensor<?x2048xf32, #encoding_result>) -> tensor<?x2048xf32, #encoding_result> attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xf8E4M3FNUZ, #encoding_lhs>, tensor<?x2048xf8E4M3FNUZ, #encoding_rhs>)
      outs(%arg2 : tensor<?x2048xf32, #encoding_result>)
      -> tensor<?x2048xf32, #encoding_result>
  return %0 : tensor<?x2048xf32, #encoding_result>
}
// CHECK-LABEL: matmul_f8_f8_f32_large_lowering_ukernel_provider
// CHECK:      iree_codegen.inner_tiled
// CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:     kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ,  intrinsics_m = 8, subgroups_m = 2, intrinsics_n = 4, subgroups_n = 4>
