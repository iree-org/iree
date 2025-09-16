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

func.func @matmul_lowering_ukernel_provider() attributes {hal.executable.target = #executable_target_rocm_hsaco_fb} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf16, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf16, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf16, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
      -> tensor<?x?xf32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf16, #encoding_lhs>,
                   tensor<?x?xf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  return
}
// CHECK-LABEL: matmul_lowering_ukernel_provider
// CHECK:      iree_codegen.inner_tiled
// CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:     kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x16_F16, intrinsics_m = 8, subgroups_m = 2, intrinsics_n = 4, subgroups_n = 4>
