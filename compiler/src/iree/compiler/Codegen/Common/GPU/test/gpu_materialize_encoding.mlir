// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-materialize-device-encoding))" \
// RUN:   --iree-gpu-test-target=gfx942 \
// RUN:   --split-input-file %s | FileCheck %s

//-----------------------------------------------------------------------------
// 1. MFMA_F32_16x16x4_F32
//-----------------------------------------------------------------------------

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], original_type = tensor<255x513xf32>,
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 16>>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
func.func @set_encoding_LHS() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xf32, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xf32>> -> tensor<255x513xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<255x513xf32,  #encoding>>
  return
}

// CHECK-LABEL: func.func @set_encoding_LHS
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<16x129x4x16xf32>
// CHECK:         %[[PACK:.*]] = tensor.pack %{{.+}} padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [1, 0]
// CHECK-SAME:      inner_tiles = [4, 16] into %[[EMPTY]]
// CHECK-SAME:      : tensor<255x513xf32> -> tensor<16x129x4x16xf32>
// CHECK:         flow.dispatch.tensor.store %[[PACK]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], original_type = tensor<255x513xf32>,
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 16>>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
func.func @set_encoding_RHS() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xf32, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xf32>> -> tensor<255x513xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<255x513xf32,  #encoding>>
  return
}

// CHECK-LABEL: func.func @set_encoding_RHS
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<33x64x4x16xf32>
// CHECK:         %[[PACK:.*]] = tensor.pack %{{.+}} padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [1, 0]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [4, 16] into %[[EMPTY]]
// CHECK-SAME:      : tensor<255x513xf32> -> tensor<33x64x4x16xf32>
// CHECK:         flow.dispatch.tensor.store %[[PACK]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], original_type = tensor<255x513xf32>,
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 16>>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
func.func @set_encoding_ACC() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xf32, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xf32>> -> tensor<255x513xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<255x513xf32,  #encoding>>
  return
}

// CHECK-LABEL: func.func @set_encoding_ACC
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<16x33x16x16xf32>
// CHECK:         %[[PACK:.*]] = tensor.pack %{{.+}} padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [16, 16] into %[[EMPTY]]
// CHECK-SAME:      : tensor<255x513xf32> -> tensor<16x33x16x16xf32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK:         %[[EMPTY2:.*]] = tensor.empty() : tensor<16x33x4x16x4xf32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:      ins(%[[EXPAND]]
// CHECK-SAME:      outs(%[[EMPTY2]]
// CHECK-SAME:      permutation = [0, 1, 2, 4, 3]
// CHECK:         flow.dispatch.tensor.store %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], original_type = tensor<255x513xf32>,
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 16>>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
func.func @unset_encoding_LHS() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xf32, #encoding>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xf32, #encoding>> -> tensor<255x513xf32, #encoding>
  %3 = iree_encoding.unset_encoding %2 : tensor<255x513xf32, #encoding> -> tensor<255x513xf32>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xf32> -> !flow.dispatch.tensor<writeonly:tensor<255x513xf32>>
  return
}

// CHECK-LABEL: func.func @unset_encoding_LHS() {
// CHECK:         %[[UNSET_EMPTY:.*]] = tensor.empty() : tensor<255x513xf32>
// CHECK:         tensor.unpack {{.+}} outer_dims_perm = [0, 1] inner_dims_pos = [1, 0] inner_tiles = [4, 16]
// CHECK-SAME:       tensor<16x129x4x16xf32> -> tensor<255x513xf32>

// -----

#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], original_type = tensor<255x513xf32>,
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 16>>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
func.func @unset_encoding_RHS() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xf32, #encoding>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xf32, #encoding>> -> tensor<255x513xf32, #encoding>
  %3 = iree_encoding.unset_encoding %2 : tensor<255x513xf32, #encoding> -> tensor<255x513xf32>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xf32> -> !flow.dispatch.tensor<writeonly:tensor<255x513xf32>>
  return
}

// CHECK-LABEL: func.func @unset_encoding_RHS() {
// CHECK:         %[[UNSET_EMPTY:.*]] = tensor.empty() : tensor<255x513xf32>
// CHECK:         tensor.unpack {{.+}} outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [4, 16]
// CHECK-SAME:      tensor<33x64x4x16xf32> -> tensor<255x513xf32>

// -----

#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], original_type = tensor<255x513xf32>,
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    round_dims_to = array<i64: 16, 16, 16>>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
func.func @unset_encoding_ACC() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<255x513xf32, #encoding>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<255x513xf32>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<255x513xf32, #encoding>> -> tensor<255x513xf32, #encoding>
  %3 = iree_encoding.unset_encoding %2 : tensor<255x513xf32, #encoding> -> tensor<255x513xf32>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xf32> -> !flow.dispatch.tensor<writeonly:tensor<255x513xf32>>
  return
}

// CHECK-LABEL: func.func @unset_encoding_ACC() {
// CHECK:         %[[UNSET_EMPTY:.*]] = tensor.empty() : tensor<16x33x4x4x16xf32>
// CHECK:         %[[UNSET_TRANSPOSE:.*]] = linalg.transpose ins(%{{.+}} : tensor<16x33x4x16x4xf32>)
// CHECK-SAME:       outs(%[[UNSET_EMPTY]] : tensor<16x33x4x4x16xf32>) permutation = [0, 1, 2, 4, 3]
// CHECK:         %[[UNSET_COLLAPSE:.*]] = tensor.collapse_shape %[[UNSET_TRANSPOSE]]
// CHECK-SAME:      tensor<16x33x4x4x16xf32> into tensor<16x33x16x16xf32>
// CHECK:         %[[UNSET_EMPTY:.*]] = tensor.empty() : tensor<255x513xf32>
// CHECK:         tensor.unpack %[[UNSET_COLLAPSE:.*]] outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into
// CHECK-SAME:      tensor<16x33x16x16xf32> -> tensor<255x513xf32>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 16, 16, 16>>
#pipeline_layout = #hal.pipeline.layout<push_constants = 3, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
func.func @matmul_lowering_f32f32f32() {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0)
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
// CHECK:     func.func @matmul_lowering_f32f32f32
// CHECK-DAG:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(0)
// CHECK-DAG:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(1)
// CHECK-DAG:   %[[ACC_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(2)
// CHECK-DAG:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]{{.+}} -> tensor<?x?x4x16xf32>
// CHECK-DAG:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]{{.+}} -> tensor<?x?x4x16xf32>
// CHECK-DAG:   %[[ACC:.+]] = flow.dispatch.tensor.load %[[ACC_BINDING]]{{.+}} -> tensor<?x?x4x16x4xf32>
// CHECK:       %[[MMA:.+]] = iree_gpu.multi_mma %[[LHS]], %[[RHS]], %[[ACC]]
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:    iterator_types = [#iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<parallel>, #iree_gpu.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, unroll_m = 1, unroll_n = 1, unroll_k = 1>
// CHECK:       flow.dispatch.tensor.store %[[MMA]], %[[ACC_BINDING]]
