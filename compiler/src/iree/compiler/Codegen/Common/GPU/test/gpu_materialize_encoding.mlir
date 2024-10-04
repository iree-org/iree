// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-materialize-device-encoding))" \
// RUN:   --iree-gpu-test-target=gfx942 \
// RUN:   --split-input-file %s | FileCheck %s

//-----------------------------------------------------------------------------
// 1. MFMA_F32_16x16x4_F32
//-----------------------------------------------------------------------------

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 16>>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @empty_fill_encoding_unroll8x8x4_MFMA_F32_16x16x4_F32() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xf32, #encoding>>
  %cst = arith.constant 0.0 : f32
  %1 = tensor.empty() : tensor<255x513xf32, #encoding>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<255x513xf32, #encoding>) -> tensor<255x513xf32, #encoding>
  flow.dispatch.tensor.store %2, %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<255x513xf32, #encoding>>
  return
}
// CHECK-LABEL: func.func @empty_fill_encoding_unroll8x8x4_MFMA_F32_16x16x4_F32
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<2x33x8x4x16x4xf32>
// CHECK:         %{{.+}} = linalg.fill ins({{.+}}) outs(%[[EMPTY]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 16>>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @set_encoding_LHS_unroll8x8x4_MFMA_F32_16x16x4_F32() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xf32, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xf32>> -> tensor<255x513xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<255x513xf32,  #encoding>>
  return
}

// CHECK-LABEL: func.func @set_encoding_LHS_unroll8x8x4_MFMA_F32_16x16x4_F32
// CHECK:         %[[PACK:.*]] = tensor.pack %{{.+}} padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 16]
// CHECK-SAME:      : tensor<255x513xf32> -> tensor<2x33x128x16xf32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<2x33x128x16xf32> into tensor<2x33x8x16x4x4xf32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<2x33x8x16x4x4xf32>)
// CHECK-SAME:       outs({{.*}} : tensor<2x33x8x4x16x4xf32>)
// CHECK-SAME:       permutation = [0, 1, 2, 5, 3, 4]
// CHECK:         flow.dispatch.tensor.store %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 16>>
#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @set_encoding_LHS_dynamic_unroll8x8x4_MFMA_F32_16x16x4_F32() {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding>>{%M, %K}
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%M, %K}
      -> tensor<?x?xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : tensor<?x?xf32, #encoding>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding>>{%M, %K}
  return
}
// CHECK-LABEL: func.func @set_encoding_LHS_dynamic_unroll8x8x4_MFMA_F32_16x16x4_F32
// CHECK:         %[[PACK:.*]] = tensor.pack %{{.+}} padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 16]
// CHECK-SAME:      : tensor<?x?xf32> -> tensor<?x?x128x16xf32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<?x?x128x16xf32> into tensor<?x?x8x16x4x4xf32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<?x?x8x16x4x4xf32>)
// CHECK-SAME:       outs({{.*}} : tensor<?x?x8x4x16x4xf32>)
// CHECK-SAME:       permutation = [0, 1, 2, 5, 3, 4]
// CHECK:         flow.dispatch.tensor.store %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 16>>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @set_encoding_RHS_unroll8x8x4_MFMA_F32_16x16x4_F32() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xf32, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xf32>> -> tensor<255x513xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<255x513xf32,  #encoding>>
  return
}

// CHECK-LABEL: func.func @set_encoding_RHS_unroll8x8x4_MFMA_F32_16x16x4_F32
// CHECK:         %[[PACK:.*]] = tensor.pack %{{.+}} padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [1, 0]
// CHECK-SAME:      inner_dims_pos = [1, 0]
// CHECK-SAME:      inner_tiles = [128, 16]
// CHECK-SAME:      : tensor<255x513xf32> -> tensor<5x16x128x16xf32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<5x16x128x16xf32> into tensor<5x16x4x2x16x4x4xf32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<5x16x4x2x16x4x4xf32>)
// CHECK-SAME:       outs({{.*}} : tensor<5x16x4x2x4x16x4xf32>)
// CHECK-SAME:       permutation = [0, 1, 2, 3, 6, 4, 5]
// CHECK:         flow.dispatch.tensor.store %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 16>>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @set_encoding_ACC_unroll8x8x4_MFMA_F32_16x16x4_F32() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xf32, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xf32>> -> tensor<255x513xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<255x513xf32,  #encoding>>
  return
}

// CHECK-LABEL: func.func @set_encoding_ACC_unroll8x8x4_MFMA_F32_16x16x4_F32
// CHECK:         %[[PACK:.*]] = tensor.pack %{{.+}} padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 128]
// CHECK-SAME:      : tensor<255x513xf32> -> tensor<2x5x128x128xf32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<2x5x128x128xf32> into tensor<2x5x8x4x4x4x2x16xf32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<2x5x8x4x4x4x2x16xf32>)
// CHECK-SAME:       outs({{.*}} : tensor<2x5x8x4x2x4x16x4xf32>)
// CHECK-SAME:       permutation = [0, 1, 2, 5, 6, 3, 7, 4]
// CHECK:         flow.dispatch.tensor.store %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 16>>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @unset_encoding_ACC_unroll8x8x4_MFMA_F32_16x16x4_F32() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xf32, #encoding>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xf32, #encoding>> -> tensor<255x513xf32, #encoding>
  %3 = iree_encoding.unset_encoding %2 : tensor<255x513xf32, #encoding> -> tensor<255x513xf32>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xf32> -> !flow.dispatch.tensor<writeonly:tensor<255x513xf32>>
  return
}

// CHECK-LABEL: func.func @unset_encoding_ACC_unroll8x8x4_MFMA_F32_16x16x4_F32() {
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%{{.+}} : tensor<2x5x8x4x2x4x16x4xf32>)
// CHECK-SAME:       outs({{.*}} : tensor<2x5x8x4x4x4x2x16xf32>)
// CHECK-SAME:       permutation = [0, 1, 2, 5, 7, 3, 4, 6]
// CHECK:         %[[COLLAPSE:.*]] = tensor.collapse_shape %[[TRANSPOSE]]
// CHECK-SAME:      : tensor<2x5x8x4x4x4x2x16xf32> into tensor<2x5x128x128xf32>
// CHECK:         %[[UNPACK:.*]] = tensor.unpack %[[COLLAPSE]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 128]
// CHECK-SAME:      : tensor<2x5x128x128xf32> -> tensor<255x513xf32>
// CHECK:         flow.dispatch.tensor.store %[[UNPACK]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 16>>
#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @unset_encoding_ACC_dynamic_unroll8x8x4_MFMA_F32_16x16x4_F32() {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%M, %K}
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%M, %K}
      -> tensor<?x?xf32, #encoding>
  %3 = iree_encoding.unset_encoding %2 : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : tensor<?x?xf32>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%M, %K}
  return
}
// CHECK-LABEL: func.func @unset_encoding_ACC_dynamic_unroll8x8x4_MFMA_F32_16x16x4_F32
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%{{.+}} : tensor<?x?x8x4x2x4x16x4xf32>)
// CHECK-SAME:       outs({{.*}} : tensor<?x?x8x4x4x4x2x16xf32>)
// CHECK-SAME:       permutation = [0, 1, 2, 5, 7, 3, 4, 6]
// CHECK:         %[[COLLAPSE:.*]] = tensor.collapse_shape %[[TRANSPOSE]]
// CHECK-SAME:      : tensor<?x?x8x4x4x4x2x16xf32> into tensor<?x?x128x128xf32>
// CHECK:         %[[UNPACK:.*]] = tensor.unpack %[[COLLAPSE]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 128]
// CHECK-SAME:      : tensor<?x?x128x128xf32> -> tensor<?x?xf32>
// CHECK:         flow.dispatch.tensor.store %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#pipeline_layout_3 = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @matmul_lowering_unroll8x8x4_MFMA_F32_16x16x4_F32() {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xf32, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xf32, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xf32, #encoding_result>>{%M, %N}
      -> tensor<?x?xf32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
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
// CHECK:     func.func @matmul_lowering_unroll8x8x4_MFMA_F32_16x16x4_F32
// CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(0)
// CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(1)
// CHECK-DAG:   %[[ACC_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(2)
// CHECK-DAG:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]{{.+}} -> tensor<?x?x8x4x16x4xf32>
// CHECK-DAG:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]{{.+}} -> tensor<?x?x4x2x4x16x4xf32>
// CHECK-DAG:   %[[ACC:.+]] = flow.dispatch.tensor.load %[[ACC_BINDING]]{{.+}} -> tensor<?x?x8x4x2x4x16x4xf32>
// CHECK:       %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS]], %[[RHS]], %[[ACC]]
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, unroll_m = 8, unroll_n = 2, unroll_n_to_subgroups = 4, unroll_k = 4>
// CHECK:       flow.dispatch.tensor.store %[[MMA]], %[[ACC_BINDING]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#pipeline_layout_4 = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @batch_matmul_lowering_unroll8x8x4_MFMA_F32_16x16x4_F32() {
  %c0 = arith.constant 0 : index
  %B = hal.interface.constant.load layout(#pipeline_layout_4) ordinal(0) : index
  %M = hal.interface.constant.load layout(#pipeline_layout_4) ordinal(1) : index
  %N = hal.interface.constant.load layout(#pipeline_layout_4) ordinal(2) : index
  %K = hal.interface.constant.load layout(#pipeline_layout_4) ordinal(3) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_4) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32, #encoding_lhs>>{%B, %M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_4) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32, #encoding_rhs>>{%B, %K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_4) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?x?xf32, #encoding_result>>{%B, %M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [%B, %M, %K], strides = [1, 1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32, #encoding_lhs>>{%B, %M, %K}
      -> tensor<?x?x?xf32, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [%B, %K, %N], strides = [1, 1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?x?xf32, #encoding_rhs>>{%B, %K, %N}
      -> tensor<?x?x?xf32, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [%B, %M, %N], strides = [1, 1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?x?xf32, #encoding_result>>{%B, %M, %N}
      -> tensor<?x?x?xf32, #encoding_result>
  %6 = linalg.batch_matmul
      ins(%3, %4 : tensor<?x?x?xf32, #encoding_lhs>,
                   tensor<?x?x?xf32, #encoding_rhs>)
      outs(%5 : tensor<?x?x?xf32, #encoding_result>)
      -> tensor<?x?x?xf32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [%B, %M, %N], strides = [1, 1, 1]
      : tensor<?x?x?xf32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?x?xf32, #encoding_result>>{%B, %M, %N}
  return
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK:     func.func @batch_matmul_lowering_unroll8x8x4_MFMA_F32_16x16x4_F32
// CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(0)
// CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(1)
// CHECK-DAG:   %[[ACC_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(2)
// CHECK-DAG:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]{{.+}} -> tensor<?x?x?x8x4x16x4xf32>
// CHECK-DAG:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]{{.+}} -> tensor<?x?x?x4x2x4x16x4xf32>
// CHECK-DAG:   %[[ACC:.+]] = flow.dispatch.tensor.load %[[ACC_BINDING]]{{.+}} -> tensor<?x?x?x8x4x2x4x16x4xf32>
// CHECK:       %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS]], %[[RHS]], %[[ACC]]
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, unroll_m = 8, unroll_n = 2, unroll_n_to_subgroups = 4, unroll_k = 4>
// CHECK:       flow.dispatch.tensor.store %[[MMA]], %[[ACC_BINDING]]

//-----------------------------------------------------------------------------
// 2. MFMA_I32_16x16x32_I8
//-----------------------------------------------------------------------------

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 32>>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @set_encoding_LHS_unroll8x8x2_MFMA_I32_16x16x32_I8() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xi8>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xi8, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xi8>> -> tensor<255x513xi8>
  %3 = iree_encoding.set_encoding %2 : tensor<255x513xi8> -> tensor<255x513xi8, #encoding>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xi8, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<255x513xi8,  #encoding>>
  return
}

// CHECK-LABEL: func.func @set_encoding_LHS_unroll8x8x2_MFMA_I32_16x16x32_I8
// CHECK:         %[[PACK:.*]] = tensor.pack %{{.+}} padding_value(%{{.+}} : i8)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 64]
// CHECK-SAME:      : tensor<255x513xi8> -> tensor<2x9x128x64xi8>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<2x9x128x64xi8> into tensor<2x9x8x16x2x4x8xi8>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<2x9x8x16x2x4x8xi8>)
// CHECK-SAME:       outs({{.*}} : tensor<2x9x8x4x16x2x8xi8>)
// CHECK-SAME:       permutation = [0, 1, 2, 5, 3, 4, 6]
// CHECK:         flow.dispatch.tensor.store %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 32>>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @set_encoding_RHS_unroll8x8x2_MFMA_I32_16x16x32_I8() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xi8>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xi8, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xi8>> -> tensor<255x513xi8>
  %3 = iree_encoding.set_encoding %2 : tensor<255x513xi8> -> tensor<255x513xi8, #encoding>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xi8, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<255x513xi8,  #encoding>>
  return
}

// CHECK-LABEL: func.func @set_encoding_RHS_unroll8x8x2_MFMA_I32_16x16x32_I8
// CHECK:         %[[PACK:.*]] = tensor.pack %{{.+}} padding_value(%{{.+}} : i8)
// CHECK-SAME:      outer_dims_perm = [1, 0]
// CHECK-SAME:      inner_dims_pos = [1, 0]
// CHECK-SAME:      inner_tiles = [128, 64]
// CHECK-SAME:      : tensor<255x513xi8> -> tensor<5x4x128x64xi8>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<5x4x128x64xi8> into tensor<5x4x4x2x16x2x4x8xi8>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<5x4x4x2x16x2x4x8xi8>)
// CHECK-SAME:       outs({{.*}} : tensor<5x4x4x2x4x16x2x8xi8>)
// CHECK-SAME:       permutation = [0, 1, 2, 3, 6, 4, 5, 7]
// CHECK:         flow.dispatch.tensor.store %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 32>>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @set_encoding_ACC_unroll8x8x2_MFMA_I32_16x16x32_I8() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xi32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xi32, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xi32>> -> tensor<255x513xi32>
  %3 = iree_encoding.set_encoding %2 : tensor<255x513xi32> -> tensor<255x513xi32, #encoding>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xi32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<255x513xi32,  #encoding>>
  return
}

// CHECK-LABEL: func.func @set_encoding_ACC_unroll8x8x2_MFMA_I32_16x16x32_I8
// CHECK:         %[[PACK:.*]] = tensor.pack %{{.+}} padding_value(%{{.+}} : i32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 128]
// CHECK-SAME:      : tensor<255x513xi32> -> tensor<2x5x128x128xi32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<2x5x128x128xi32> into tensor<2x5x8x4x4x4x2x16xi32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<2x5x8x4x4x4x2x16xi32>)
// CHECK-SAME:       outs({{.*}} : tensor<2x5x8x4x2x4x16x4xi32>)
// CHECK-SAME:       permutation = [0, 1, 2, 5, 6, 3, 7, 4]
// CHECK:         flow.dispatch.tensor.store %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 32>>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @unset_encoding_ACC_unroll8x8x2_MFMA_I32_16x16x32_I8() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xi32, #encoding>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xi32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xi32, #encoding>> -> tensor<255x513xi32, #encoding>
  %3 = iree_encoding.unset_encoding %2 : tensor<255x513xi32, #encoding> -> tensor<255x513xi32>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xi32> -> !flow.dispatch.tensor<writeonly:tensor<255x513xi32>>
  return
}

// CHECK-LABEL: func.func @unset_encoding_ACC_unroll8x8x2_MFMA_I32_16x16x32_I8() {
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%{{.+}} : tensor<2x5x8x4x2x4x16x4xi32>)
// CHECK-SAME:       outs({{.*}} : tensor<2x5x8x4x4x4x2x16xi32>)
// CHECK-SAME:       permutation = [0, 1, 2, 5, 7, 3, 4, 6]
// CHECK:         %[[COLLAPSE:.*]] = tensor.collapse_shape %[[TRANSPOSE]]
// CHECK-SAME:      : tensor<2x5x8x4x4x4x2x16xi32> into tensor<2x5x128x128xi32>
// CHECK:         %[[UNPACK:.*]] = tensor.unpack %[[COLLAPSE]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 128]
// CHECK-SAME:      : tensor<2x5x128x128xi32> -> tensor<255x513xi32>
// CHECK:         flow.dispatch.tensor.store %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 32>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 32>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 32>>
#pipeline_layout_3 = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

func.func @matmul_lowering_unroll8x8x2_MFMA_I32_16x16x32_I8() {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout_3) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout_3) binding(2) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%M, %K], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
      -> tensor<?x?xi8, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%K, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
      -> tensor<?x?xi8, #encoding_rhs>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
      -> tensor<?x?xi32, #encoding_result>
  %6 = linalg.matmul
      ins(%3, %4 : tensor<?x?xi8, #encoding_lhs>,
                   tensor<?x?xi8, #encoding_rhs>)
      outs(%5 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [%M, %N], strides = [1, 1]
      : tensor<?x?xi32, #encoding_result>
      -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32, #encoding_result>>{%M, %N}
  return
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK:     func.func @matmul_lowering_unroll8x8x2_MFMA_I32_16x16x32_I8
// CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(0)
// CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(1)
// CHECK-DAG:   %[[ACC_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(2)
// CHECK-DAG:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]{{.+}} -> tensor<?x?x8x4x16x2x8xi8>
// CHECK-DAG:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]{{.+}} -> tensor<?x?x4x2x4x16x2x8xi8>
// CHECK-DAG:   %[[ACC:.+]] = flow.dispatch.tensor.load %[[ACC_BINDING]]{{.+}} -> tensor<?x?x8x4x2x4x16x4xi32>
// CHECK:       %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS]], %[[RHS]], %[[ACC]]
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x32_I8, unroll_m = 8, unroll_n = 2, unroll_n_to_subgroups = 4, unroll_k = 2>
// CHECK:       flow.dispatch.tensor.store %[[MMA]], %[[ACC_BINDING]]
