// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-materialize-device-encoding))" \
// RUN:   --iree-gpu-test-target=gfx90a \
// RUN:   --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#pipeline_layout_3 = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_lowering_MFMA_f32_16x16x8_bf16() {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xbf16, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xbf16, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xbf16, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
      -> tensor<?x?xf32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xbf16, #encoding_lhs>,
                   tensor<?x?xbf16, #encoding_rhs>)
      outs(%5 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  return
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK:     func.func @matmul_lowering_MFMA_f32_16x16x8_bf16
// CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(0)
// CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(1)
// CHECK-DAG:   %[[ACC_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(2)
// CHECK-DAG:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]{{.+}} -> tensor<?x?x4x4x16x4x2xbf16>
// CHECK-DAG:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]{{.+}} -> tensor<?x?x4x2x4x16x4x2xbf16>
// CHECK-DAG:   %[[ACC:.+]] = flow.dispatch.tensor.load %[[ACC_BINDING]]{{.+}} -> tensor<?x?x4x4x2x4x16x4xf32>
// CHECK:       %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS]], %[[RHS]], %[[ACC]]
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x8_BF16, unroll_m = 4, unroll_n = 2, subgroups_n = 4, unroll_k = 4>
// CHECK:       flow.dispatch.tensor.store %[[MMA]], %[[ACC_BINDING]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f64, f64, f64], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f64, f64, f64], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f64, f64, f64], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#pipeline_layout_3 = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_lowering_MFMA_f64_16x16x4_f64() {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf64, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf64, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf64, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf64, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf64, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf64, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf64, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf64, #encoding_result>>{%M, %N}
      -> tensor<?x?xf64, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf64, #encoding_lhs>,
                   tensor<?x?xf64, #encoding_rhs>)
      outs(%5 : tensor<?x?xf64, #encoding_result>)
      -> tensor<?x?xf64, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xf64, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf64, #encoding_result>>{%M, %N}
  return
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK:     func.func @matmul_lowering_MFMA_f64_16x16x4_f64
// CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(0)
// CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(1)
// CHECK-DAG:   %[[ACC_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(2)
// CHECK-DAG:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]{{.+}} -> tensor<?x?x4x4x16x2xf64>
// CHECK-DAG:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]{{.+}} -> tensor<?x?x4x4x16x2xf64>
// CHECK-DAG:   %[[ACC:.+]] = flow.dispatch.tensor.load %[[ACC_BINDING]]{{.+}} -> tensor<?x?x4x4x4x4x16xf64>
// CHECK:       %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS]], %[[RHS]], %[[ACC]]
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F64_16x16x4_F64, unroll_m = 4, subgroups_n = 4, unroll_k = 2>
// CHECK:       flow.dispatch.tensor.store %[[MMA]], %[[ACC_BINDING]]
