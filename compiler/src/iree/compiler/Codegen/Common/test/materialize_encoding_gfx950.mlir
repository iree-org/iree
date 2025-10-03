// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding{test-cl-gpu-target}))" \
// RUN:   --iree-gpu-test-target=gfx950 \
// RUN:   --split-input-file %s | FileCheck %s

// Contains tests that differ from gfx942/MI-300

//-----------------------------------------------------------------------------
// 1. MFMA_I32_16x16x64_I8
//-----------------------------------------------------------------------------

// XXX I don't think these test names are correct but I don't understand what "unroll" has to do with anything XXX
#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [255, 513, ?]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @set_encoding_LHS_unroll8x8x2_MFMA_I32_16x16x64_I8() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<255x513xi8>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<255x513xi8, #encoding>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<255x513xi8>> -> tensor<255x513xi8>
  %3 = iree_encoding.set_encoding %2 : tensor<255x513xi8> -> tensor<255x513xi8, #encoding>
  iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xi8, #encoding> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<255x513xi8,  #encoding>>
  return
}

// CHECK-LABEL: func.func @set_encoding_LHS_unroll8x8x2_MFMA_I32_16x16x64_I8
// CHECK:         %[[PACK:.*]] = linalg.pack %{{.+}} padding_value(%{{.+}} : i8)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 64]
// CHECK-SAME:      : tensor<255x513xi8> -> tensor<2x9x128x64xi8>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<2x9x128x64xi8> into tensor<2x9x4x8x4x4x16xi8>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<2x9x4x8x4x4x16xi8>)
// CHECK-SAME:       outs({{.*}} : tensor<2x9x8x4x4x4x16xi8>)
// CHECK-SAME:       permutation = [0, 1, 3, 5, 2, 4, 6]
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [255, 513, ?]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @set_encoding_RHS_unroll8x8x2_MFMA_I32_16x16x64_I8() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<255x513xi8>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<255x513xi8, #encoding>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<255x513xi8>> -> tensor<255x513xi8>
  %3 = iree_encoding.set_encoding %2 : tensor<255x513xi8> -> tensor<255x513xi8, #encoding>
  iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xi8, #encoding> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<255x513xi8,  #encoding>>
  return
}

// CHECK-LABEL: func.func @set_encoding_RHS_unroll8x8x2_MFMA_I32_16x16x64_I8
// CHECK:         %[[PACK:.*]] = linalg.pack %{{.+}} padding_value(%{{.+}} : i8)
// CHECK-SAME:      outer_dims_perm = [1, 0]
// CHECK-SAME:      inner_dims_pos = [1, 0]
// CHECK-SAME:      inner_tiles = [128, 64]
// CHECK-SAME:      : tensor<255x513xi8> -> tensor<5x4x128x64xi8>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<5x4x128x64xi8> into tensor<5x4x4x16x2x4x16xi8>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<5x4x4x16x2x4x16xi8>)
// CHECK-SAME:       outs({{.*}} : tensor<5x4x4x2x4x16x16xi8>)
// CHECK-SAME:       permutation = [0, 1, 2, 4, 5, 3, 6]
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [255, 513, ?]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @set_encoding_ACC_unroll8x8x2_MFMA_I32_16x16x64_I8() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<255x513xi32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<255x513xi32, #encoding>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<255x513xi32>> -> tensor<255x513xi32>
  %3 = iree_encoding.set_encoding %2 : tensor<255x513xi32> -> tensor<255x513xi32, #encoding>
  iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xi32, #encoding> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<255x513xi32,  #encoding>>
  return
}

// CHECK-LABEL: func.func @set_encoding_ACC_unroll8x8x2_MFMA_I32_16x16x64_I8
// CHECK:         %[[PACK:.*]] = linalg.pack %{{.+}} padding_value(%{{.+}} : i32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 128]
// CHECK-SAME:      : tensor<255x513xi32> -> tensor<2x5x128x128xi32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<2x5x128x128xi32> into tensor<2x5x4x8x4x4x16x2xi32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<2x5x4x8x4x4x16x2xi32>)
// CHECK-SAME:       outs({{.*}} : tensor<2x5x4x8x2x4x16x4xi32>)
// CHECK-SAME:       permutation = [0, 1, 5, 3, 7, 2, 6, 4]
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [255, 513, ?]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @unset_encoding_ACC_unroll8x8x2_MFMA_I32_16x16x64_I8() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<255x513xi32, #encoding>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<255x513xi32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<255x513xi32, #encoding>> -> tensor<255x513xi32, #encoding>
  %3 = iree_encoding.unset_encoding %2 : tensor<255x513xi32, #encoding> -> tensor<255x513xi32>
  iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<255x513xi32>>
  return
}

// CHECK-LABEL: func.func @unset_encoding_ACC_unroll8x8x2_MFMA_I32_16x16x64_I8() {
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%{{.+}} : tensor<2x5x4x8x2x4x16x4xi32>)
// CHECK-SAME:       outs({{.*}} : tensor<2x5x4x8x4x4x16x2xi32>)
// CHECK-SAME:       permutation = [0, 1, 5, 3, 7, 2, 6, 4]
// CHECK:         %[[COLLAPSE:.*]] = tensor.collapse_shape %[[TRANSPOSE]]
// CHECK-SAME:      : tensor<2x5x4x8x4x4x16x2xi32> into tensor<2x5x128x128xi32>
// CHECK:         %[[UNPACK:.*]] = linalg.unpack %[[COLLAPSE]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 128]
// CHECK-SAME:      : tensor<2x5x128x128xi32> -> tensor<255x513xi32>
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#pipeline_layout_3 = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @matmul_lowering_MFMA_I32_16x16x64_I8() {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi8, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi8, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK:     func.func @matmul_lowering_MFMA_I32_16x16x64_I8
// CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(0)
// CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(1)
// CHECK-DAG:   %[[ACC_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(2)
// CHECK-DAG:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]{{.+}} -> tensor<?x?x8x4x4x4x16xi8>
// CHECK-DAG:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]{{.+}} -> tensor<?x?x4x2x4x16x16xi8>
// CHECK-DAG:   %[[ACC:.+]] = iree_tensor_ext.dispatch.tensor.load %[[ACC_BINDING]]{{.+}} -> tensor<?x?x4x8x2x4x16x4xi32>
// CHECK:       %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x64_I8, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4>
// CHECK:       iree_tensor_ext.dispatch.tensor.store %[[MMA]], %[[ACC_BINDING]]

// -----

//---------------------------------------------------------------------------
// 3. Additional element types, testing only the multi_mma, not set_encoding.
//---------------------------------------------------------------------------

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f8E4M3FN, f8E4M3FN, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f8E4M3FN, f8E4M3FN, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f8E4M3FN, f8E4M3FN, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?, ?]>
#pipeline_layout_4 = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @batch_matmul_lowering_MFMA_F32_16x16x128_F8E4M3FN() {
  %c0 = arith.constant 0 : index
  %B = hal.interface.constant.load layout(#pipeline_layout_4) ordinal(0) : index
  %M = hal.interface.constant.load layout(#pipeline_layout_4) ordinal(1) : index
  %N = hal.interface.constant.load layout(#pipeline_layout_4) ordinal(2) : index
  %K = hal.interface.constant.load layout(#pipeline_layout_4) ordinal(3) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_4) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xf8E4M3FN, #encoding_lhs>>{%B, %M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_4) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xf8E4M3FN, #encoding_rhs>>{%B, %K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_4) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?xf32, #encoding_result>>{%B, %M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [%B, %M, %K], strides = [1, 1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xf8E4M3FN, #encoding_lhs>>{%B, %M, %K}
      -> tensor<?x?x?xf8E4M3FN, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [%B, %K, %N], strides = [1, 1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xf8E4M3FN, #encoding_rhs>>{%B, %K, %N}
      -> tensor<?x?x?xf8E4M3FN, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [%B, %M, %N], strides = [1, 1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?xf32, #encoding_result>>{%B, %M, %N}
      -> tensor<?x?x?xf32, #encoding_result>
  %6 = linalg.batch_matmul
      ins(%3, %4 : tensor<?x?x?xf8E4M3FN, #encoding_lhs>,
                   tensor<?x?x?xf8E4M3FN, #encoding_rhs>)
      outs(%5 : tensor<?x?x?xf32, #encoding_result>)
      -> tensor<?x?x?xf32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [%B, %M, %N], strides = [1, 1, 1]
      : tensor<?x?x?xf32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?xf32, #encoding_result>>{%B, %M, %N}
  return
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK:     func.func @batch_matmul_lowering_MFMA_F32_16x16x128_F8E4M3FN
// CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(0)
// CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(1)
// CHECK-DAG:   %[[ACC_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(2)
// CHECK-DAG:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]{{.+}} -> tensor<?x?x?x8x4x4x4x32xf8E4M3FN>
// CHECK-DAG:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]{{.+}} -> tensor<?x?x?x4x2x4x16x32xf8E4M3FN>
// CHECK-DAG:   %[[ACC:.+]] = iree_tensor_ext.dispatch.tensor.load %[[ACC_BINDING]]{{.+}} -> tensor<?x?x?x4x8x2x4x16x4xf32>
// CHECK:       %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x128_F8E4M3FN, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4>
// CHECK:       iree_tensor_ext.dispatch.tensor.store %[[MMA]], %[[ACC_BINDING]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>
#pipeline_layout_4 = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @batch_matmul_lowering_MFMA_F32_16x16x32_BF16() {
  %c0 = arith.constant 0 : index
  %B = hal.interface.constant.load layout(#pipeline_layout_4) ordinal(0) : index
  %M = hal.interface.constant.load layout(#pipeline_layout_4) ordinal(1) : index
  %N = hal.interface.constant.load layout(#pipeline_layout_4) ordinal(2) : index
  %K = hal.interface.constant.load layout(#pipeline_layout_4) ordinal(3) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_4) binding(0) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xbf16, #encoding_lhs>>{%B, %M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_4) binding(1) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xbf16, #encoding_rhs>>{%B, %K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_4) binding(2) alignment(64) offset(%c0)
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?xf32, #encoding_result>>{%B, %M, %N}
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [%B, %M, %K], strides = [1, 1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xbf16, #encoding_lhs>>{%B, %M, %K}
      -> tensor<?x?x?xbf16, #encoding_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [%B, %K, %N], strides = [1, 1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x?xbf16, #encoding_rhs>>{%B, %K, %N}
      -> tensor<?x?x?xbf16, #encoding_rhs>
  %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [%B, %M, %N], strides = [1, 1, 1]
      : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?xf32, #encoding_result>>{%B, %M, %N}
      -> tensor<?x?x?xf32, #encoding_result>
  %6 = linalg.batch_matmul
      ins(%3, %4 : tensor<?x?x?xbf16, #encoding_lhs>,
                   tensor<?x?x?xbf16, #encoding_rhs>)
      outs(%5 : tensor<?x?x?xf32, #encoding_result>)
      -> tensor<?x?x?xf32, #encoding_result>
  iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [%B, %M, %N], strides = [1, 1, 1]
      : tensor<?x?x?xf32, #encoding_result>
      -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x?xf32, #encoding_result>>{%B, %M, %N}
  return
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK:     func.func @batch_matmul_lowering_MFMA_F32_16x16x32_BF16
// CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(0)
// CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(1)
// CHECK-DAG:   %[[ACC_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(2)
// CHECK-DAG:   %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]{{.+}} -> tensor<?x?x?x8x4x4x4x8xbf16>
// CHECK-DAG:   %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]{{.+}} -> tensor<?x?x?x4x2x4x16x8xbf16>
// CHECK-DAG:   %[[ACC:.+]] = iree_tensor_ext.dispatch.tensor.load %[[ACC_BINDING]]{{.+}} -> tensor<?x?x?x4x8x2x4x16x4xf32>
// CHECK:       %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x32_BF16, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4>
// CHECK:       iree_tensor_ext.dispatch.tensor.store %[[MMA]], %[[ACC_BINDING]]

// -----

//-----------------------------------------------------------------------------
// 1. Scaled MFMA
//-----------------------------------------------------------------------------

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding = #iree_encoding.encoding<
  operand_index = 0 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#map, #map1, #map2, #map3, #map4],
  iteration_sizes = [255, 513, 127, 32]>

func.func @set_encoding_LHS_scaled_matmul_f4_f4_f8_f8_f32(%arg0: tensor<255x127x32xf4E2M1FN>) -> tensor<255x127x32xf4E2M1FN, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<255x127x32xf4E2M1FN> -> tensor<255x127x32xf4E2M1FN, #encoding>
  return %0 : tensor<255x127x32xf4E2M1FN, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_LHS_scaled_matmul_f4_f4_f8_f8_f32(
// CHECK:         %[[PACK:.*]] = linalg.pack %{{.+}} padding_value(%{{.+}} : f4E2M1FN)
// CHECK-SAME:      outer_dims_perm = [0, 1, 2]
// CHECK-SAME:      inner_dims_pos = [0, 1, 2]
// CHECK-SAME:      inner_tiles = [64, 16, 32]
// CHECK-SAME:      : tensor<255x127x32xf4E2M1FN> -> tensor<4x8x1x64x16x32xf4E2M1FN>
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME:       : tensor<4x8x1x64x16x32xf4E2M1FN> into tensor<4x8x1x4x16x4x4x32xf4E2M1FN>
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<4x8x1x4x4x4x16x32xf4E2M1FN>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPANDED]] : tensor<4x8x1x4x16x4x4x32xf4E2M1FN>)
// CHECK-SAME:       outs(%[[EMPTY]] : tensor<4x8x1x4x4x4x16x32xf4E2M1FN>)
// CHECK-SAME:       permutation = [0, 1, 2, 3, 5, 6, 4, 7]
// CHECK:         return %[[TRANSPOSE]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding = #iree_encoding.encoding<
  operand_index = 1 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#map, #map1, #map2, #map3, #map4],
  iteration_sizes = [255, 513, 127, 32]>

func.func @set_encoding_RHS_scaled_matmul_f4_f4_f8_f8_f32(%arg0: tensor<513x127x32xf4E2M1FN>) -> tensor<513x127x32xf4E2M1FN, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<513x127x32xf4E2M1FN> -> tensor<513x127x32xf4E2M1FN, #encoding>
  return %0 : tensor<513x127x32xf4E2M1FN, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_RHS_scaled_matmul_f4_f4_f8_f8_f32(
// CHECK:         %[[PACK:.*]] = linalg.pack %{{.+}} padding_value(%{{.+}} : f4E2M1FN)
// CHECK-SAME:      outer_dims_perm = [0, 1, 2]
// CHECK-SAME:      inner_dims_pos = [0, 1, 2]
// CHECK-SAME:      inner_tiles = [128, 16, 32]
// CHECK-SAME:      : tensor<513x127x32xf4E2M1FN> -> tensor<5x8x1x128x16x32xf4E2M1FN>
// CHECK:         %[[EXPANDED:.+]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME:       : tensor<5x8x1x128x16x32xf4E2M1FN> into tensor<5x8x1x4x2x16x4x4x32xf4E2M1FN>
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<5x8x1x4x2x4x4x16x32xf4E2M1FN>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPANDED]] : tensor<5x8x1x4x2x16x4x4x32xf4E2M1FN>)
// CHECK-SAME:       outs(%[[EMPTY]] : tensor<5x8x1x4x2x4x4x16x32xf4E2M1FN>)
// CHECK-SAME:       permutation = [0, 1, 2, 3, 4, 6, 7, 5, 8]
// CHECK:         return %[[TRANSPOSE]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding = #iree_encoding.encoding<
  operand_index = 2 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#map, #map1, #map2, #map3, #map4],
  iteration_sizes = [255, 513, 127, 32]>

func.func @set_encoding_LHS_SCALES_scaled_matmul_f4_f4_f8_f8_f32(%arg0: tensor<255x127xf8E8M0FNU>) -> tensor<255x127xf8E8M0FNU, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<255x127xf8E8M0FNU> -> tensor<255x127xf8E8M0FNU, #encoding>
  return %0 : tensor<255x127xf8E8M0FNU, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_LHS_SCALES_scaled_matmul_f4_f4_f8_f8_f32(
// CHECK:         %[[PACK:.*]] = linalg.pack %{{.+}} padding_value(%{{.+}} : f8E8M0FNU)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [64, 16]
// CHECK-SAME:      : tensor<255x127xf8E8M0FNU> -> tensor<4x8x64x16xf8E8M0FNU>
// CHECK:         %[[EXPANDED:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME:       : tensor<4x8x64x16xf8E8M0FNU> into tensor<4x8x4x16x4x4xf8E8M0FNU>
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<4x8x4x4x16x4xf8E8M0FNU>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPANDED]] : tensor<4x8x4x16x4x4xf8E8M0FNU>)
// CHECK-SAME:       outs(%[[EMPTY]] : tensor<4x8x4x4x16x4xf8E8M0FNU>)
// CHECK-SAME:       permutation = [0, 1, 2, 5, 3, 4]
// CHECK:         return %[[TRANSPOSE]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding = #iree_encoding.encoding<
  operand_index = 3 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#map, #map1, #map2, #map3, #map4],
  iteration_sizes = [255, 513, 127, 32]>

func.func @set_encoding_RHS_SCALES_scaled_matmul_f4_f4_f8_f8_f32(%arg0: tensor<513x127xf8E8M0FNU>) -> tensor<513x127xf8E8M0FNU, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<513x127xf8E8M0FNU> -> tensor<513x127xf8E8M0FNU, #encoding>
  return %0 : tensor<513x127xf8E8M0FNU, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_RHS_SCALES_scaled_matmul_f4_f4_f8_f8_f32(
// CHECK:         %[[PACK:.*]] = linalg.pack %{{.+}} padding_value(%{{.+}} : f8E8M0FNU)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 16]
// CHECK-SAME:      : tensor<513x127xf8E8M0FNU> -> tensor<5x8x128x16xf8E8M0FNU>
// CHECK:         %[[EXPANDED:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME:       : tensor<5x8x128x16xf8E8M0FNU> into tensor<5x8x4x2x16x4x4xf8E8M0FNU>
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<5x8x4x2x4x16x4xf8E8M0FNU>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPANDED]] : tensor<5x8x4x2x16x4x4xf8E8M0FNU>)
// CHECK-SAME:       outs(%[[EMPTY]] : tensor<5x8x4x2x4x16x4xf8E8M0FNU>)
// CHECK-SAME:       permutation = [0, 1, 2, 3, 6, 4, 5]
// CHECK:         return %[[TRANSPOSE]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding = #iree_encoding.encoding<
  operand_index = 4 : index, op_type = scaled_matmul,
  element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32],
  user_indexing_maps = [#map, #map1, #map2, #map3, #map4],
  iteration_sizes = [255, 513, 127, 32]>

func.func @set_encoding_ACC_scaled_matmul_f4_f4_f8_f8_f32(%arg0: tensor<255x513xf32>) -> tensor<255x513xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  return %0 : tensor<255x513xf32, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_ACC_scaled_matmul_f4_f4_f8_f8_f32(
// CHECK:         %[[PACK:.*]] = linalg.pack %{{.+}} padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [64, 128]
// CHECK-SAME:      : tensor<255x513xf32> -> tensor<4x5x64x128xf32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME:       : tensor<4x5x64x128xf32> into tensor<4x5x4x4x4x4x2x16xf32>
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<4x5x4x4x2x4x16x4xf32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<4x5x4x4x4x4x2x16xf32>)
// CHECK-SAME:       outs(%[[EMPTY]] : tensor<4x5x4x4x2x4x16x4xf32>)
// CHECK-SAME:       permutation = [0, 1, 5, 2, 6, 3, 7, 4]
// CHECK:         return %[[TRANSPOSE]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding_lhs = #iree_encoding.encoding<operand_index = 0 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_lhs_scales = #iree_encoding.encoding<operand_index = 2 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_rhs_scales = #iree_encoding.encoding<operand_index = 3 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_result = #iree_encoding.encoding<operand_index = 4 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>

func.func @scaled_matmul_lowering_f4_f4_f8_f8_f32(
    %arg0: tensor<?x?x32xf4E2M1FN, #encoding_lhs>,
    %arg1: tensor<?x?x32xf4E2M1FN, #encoding_rhs>,
    %arg2: tensor<?x?xf8E8M0FNU, #encoding_lhs_scales>,
    %arg3: tensor<?x?xf8E8M0FNU, #encoding_rhs_scales>,
    %arg4: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result> {
  %0 = linalg.generic {
      indexing_maps = [#map, #map1, #map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%arg0, %arg1, %arg2, %arg3
           : tensor<?x?x32xf4E2M1FN, #encoding_lhs>, tensor<?x?x32xf4E2M1FN, #encoding_rhs>,
             tensor<?x?xf8E8M0FNU, #encoding_lhs_scales>, tensor<?x?xf8E8M0FNU, #encoding_rhs_scales>)
      outs(%arg4 : tensor<?x?xf32, #encoding_result>) {
  ^bb0(%in: f4E2M1FN, %in_0: f4E2M1FN, %in_1: f8E8M0FNU, %in_2: f8E8M0FNU, %out: f32):
    %11 = arith.scaling_extf %in, %in_1 : f4E2M1FN, f8E8M0FNU to f32
    %12 = arith.scaling_extf %in_0, %in_2 : f4E2M1FN, f8E8M0FNU to f32
    %13 = arith.mulf %11, %12 : f32
    %14 = arith.addf %out, %13 : f32
    linalg.yield %14 : f32
  } -> tensor<?x?xf32, #encoding_result>
  return %0 : tensor<?x?xf32, #encoding_result>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK:     func.func @scaled_matmul_lowering_f4_f4_f8_f8_f32(
// CHECK-SAME:  %[[LHS:.+]]: tensor<?x?x1x4x4x4x16x32xf4E2M1FN>, %[[RHS:.+]]: tensor<?x?x1x4x2x4x4x16x32xf4E2M1FN>
// CHECK-SAME:  %[[LHS_SCALES:.+]]: tensor<?x?x4x4x16x4xf8E8M0FNU>, %[[RHS_SCALES:.+]]: tensor<?x?x4x2x4x16x4xf8E8M0FNU>
// CHECK-SAME:  %[[RESULT:.+]]: tensor<?x?x4x4x2x4x16x4xf32>
// CHECK:       %[[SCALED_MATMUL:.+]] = iree_codegen.inner_tiled
// CHECK-SAME:    ins(%[[LHS]], %[[RHS]], %[[LHS_SCALES]], %[[RHS_SCALES]])
// CHECK-SAME:    outs(%[[RESULT]])
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]], #[[MAP4]]],
// CHECK-SAME:    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32, intrinsics_m = 4, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 4>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding_lhs = #iree_encoding.encoding<operand_index = 0 : index, op_type = scaled_matmul, element_types = [f8E4M3FN, f8E4M3FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1 : index, op_type = scaled_matmul, element_types = [f8E4M3FN, f8E4M3FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_lhs_scales = #iree_encoding.encoding<operand_index = 2 : index, op_type = scaled_matmul, element_types = [f8E4M3FN, f8E4M3FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_rhs_scales = #iree_encoding.encoding<operand_index = 3 : index, op_type = scaled_matmul, element_types = [f8E4M3FN, f8E4M3FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_result = #iree_encoding.encoding<operand_index = 4 : index, op_type = scaled_matmul, element_types = [f8E4M3FN, f8E4M3FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>

func.func @scaled_matmul_lowering_f8_f8_f8_f8_f32(
    %arg0: tensor<?x?x32xf8E4M3FN, #encoding_lhs>,
    %arg1: tensor<?x?x32xf8E4M3FN, #encoding_rhs>,
    %arg2: tensor<?x?xf8E8M0FNU, #encoding_lhs_scales>,
    %arg3: tensor<?x?xf8E8M0FNU, #encoding_rhs_scales>,
    %arg4: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result> {
  %0 = linalg.generic {
      indexing_maps = [#map, #map1, #map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%arg0, %arg1, %arg2, %arg3
           : tensor<?x?x32xf8E4M3FN, #encoding_lhs>, tensor<?x?x32xf8E4M3FN, #encoding_rhs>,
             tensor<?x?xf8E8M0FNU, #encoding_lhs_scales>, tensor<?x?xf8E8M0FNU, #encoding_rhs_scales>)
      outs(%arg4 : tensor<?x?xf32, #encoding_result>) {
  ^bb0(%in: f8E4M3FN, %in_0: f8E4M3FN, %in_1: f8E8M0FNU, %in_2: f8E8M0FNU, %out: f32):
    %11 = arith.scaling_extf %in, %in_1 : f8E4M3FN, f8E8M0FNU to f32
    %12 = arith.scaling_extf %in_0, %in_2 : f8E4M3FN, f8E8M0FNU to f32
    %13 = arith.mulf %11, %12 : f32
    %14 = arith.addf %out, %13 : f32
    linalg.yield %14 : f32
  } -> tensor<?x?xf32, #encoding_result>
  return %0 : tensor<?x?xf32, #encoding_result>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK:     func.func @scaled_matmul_lowering_f8_f8_f8_f8_f32(
// CHECK-SAME:  %[[LHS:.+]]: tensor<?x?x1x4x4x4x16x32xf8E4M3FN>, %[[RHS:.+]]: tensor<?x?x1x4x2x4x4x16x32xf8E4M3FN>
// CHECK-SAME:  %[[LHS_SCALES:.+]]: tensor<?x?x4x4x16x4xf8E8M0FNU>, %[[RHS_SCALES:.+]]: tensor<?x?x4x2x4x16x4xf8E8M0FNU>
// CHECK-SAME:  %[[RESULT:.+]]: tensor<?x?x4x4x2x4x16x4xf32>
// CHECK:       %[[SCALED_MATMUL:.+]] = iree_codegen.inner_tiled
// CHECK-SAME:    ins(%[[LHS]], %[[RHS]], %[[LHS_SCALES]], %[[RHS_SCALES]])
// CHECK-SAME:    outs(%[[RESULT]])
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]], #[[MAP4]]],
// CHECK-SAME:    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_scaled_mma_layout<intrinsic = MFMA_SCALE_F32_16x16x128_B32, lhs_elem_type = f8E4M3FN, rhs_elem_type = f8E4M3FN, acc_elem_type = f32, intrinsics_m = 4, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 4>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

#encoding_lhs = #iree_encoding.encoding<operand_index = 0 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_lhs_scales = #iree_encoding.encoding<operand_index = 2 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_rhs_scales = #iree_encoding.encoding<operand_index = 3 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>
#encoding_result = #iree_encoding.encoding<operand_index = 4 : index, op_type = scaled_matmul, element_types = [f4E2M1FN, f4E2M1FN, f8E8M0FNU, f8E8M0FNU, f32], user_indexing_maps = [#map, #map1, #map2, #map3, #map4], iteration_sizes = [?, ?, ?, 32]>

#executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {iree_codegen.target_info = #iree_gpu.target<
    arch = "gfx950", features = "",
    wgp = <compute = fp16, storage =  b16,
           scaled_mma = [
             <intrinsic = MFMA_SCALE_F32_32x32x64_B32,
              lhs_elem_type = f4E2M1FN,
              rhs_elem_type = f4E2M1FN,
              acc_elem_type = f32>],
           subgroup =  none, subgroup_size_choices = [64],
           max_workgroup_sizes = [1024, 1024, 1024],
           max_thread_count_per_workgroup = 1024,
           max_workgroup_memory_bytes = 163840,
           max_workgroup_counts = [2147483647, 2147483647, 2147483647],
           max_load_instruction_bits = 128,
           simds_per_wgp = 4,
           vgpr_space_bits = 16384>>,
   iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>}>
func.func @scaled_matmul_lowering_f4_f4_f8_f8_f32_MFMA_SCALE_F32_32x32x64_B32(
    %arg0: tensor<?x?x32xf4E2M1FN, #encoding_lhs>,
    %arg1: tensor<?x?x32xf4E2M1FN, #encoding_rhs>,
    %arg2: tensor<?x?xf8E8M0FNU, #encoding_lhs_scales>,
    %arg3: tensor<?x?xf8E8M0FNU, #encoding_rhs_scales>,
    %arg4: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result>
    attributes { hal.executable.target = #executable_target } {
  %0 = linalg.generic {
      indexing_maps = [#map, #map1, #map2, #map3, #map4],
      iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
      ins(%arg0, %arg1, %arg2, %arg3
           : tensor<?x?x32xf4E2M1FN, #encoding_lhs>, tensor<?x?x32xf4E2M1FN, #encoding_rhs>,
             tensor<?x?xf8E8M0FNU, #encoding_lhs_scales>, tensor<?x?xf8E8M0FNU, #encoding_rhs_scales>)
      outs(%arg4 : tensor<?x?xf32, #encoding_result>) {
  ^bb0(%in: f4E2M1FN, %in_0: f4E2M1FN, %in_1: f8E8M0FNU, %in_2: f8E8M0FNU, %out: f32):
    %11 = arith.scaling_extf %in, %in_1 : f4E2M1FN, f8E8M0FNU to f32
    %12 = arith.scaling_extf %in_0, %in_2 : f4E2M1FN, f8E8M0FNU to f32
    %13 = arith.mulf %11, %12 : f32
    %14 = arith.addf %out, %13 : f32
    linalg.yield %14 : f32
  } -> tensor<?x?xf32, #encoding_result>
  return %0 : tensor<?x?xf32, #encoding_result>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
// CHECK-DAG: #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
// CHECK:     func.func @scaled_matmul_lowering_f4_f4_f8_f8_f32_MFMA_SCALE_F32_32x32x64_B32(
// CHECK-SAME:  %[[LHS:.+]]: tensor<?x?x1x4x4x2x32x32xf4E2M1FN>, %[[RHS:.+]]: tensor<?x?x1x4x4x2x32x32xf4E2M1FN>
// CHECK-SAME:  %[[LHS_SCALES:.+]]: tensor<?x?x4x2x32x4xf8E8M0FNU>, %[[RHS_SCALES:.+]]: tensor<?x?x4x2x32x4xf8E8M0FNU>
// CHECK-SAME:  %[[RESULT:.+]]: tensor<?x?x4x4x4x2x32x4xf32>
// CHECK:       %[[SCALED_MATMUL:.+]] = iree_codegen.inner_tiled
// CHECK-SAME:    ins(%[[LHS]], %[[RHS]], %[[LHS_SCALES]], %[[RHS_SCALES]])
// CHECK-SAME:    outs(%[[RESULT]])
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]], #[[MAP4]]],
// CHECK-SAME:    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>, #linalg.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_scaled_mma_layout<intrinsic = MFMA_SCALE_F32_32x32x64_B32, lhs_elem_type = f4E2M1FN, rhs_elem_type = f4E2M1FN, acc_elem_type = f32, intrinsics_m = 4, subgroups_n = 4, intrinsics_k = 4>
