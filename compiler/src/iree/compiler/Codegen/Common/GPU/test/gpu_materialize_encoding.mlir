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
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<16x129x16x4xf32>
// CHECK:         %[[PACK:.*]] = tensor.pack %{{.+}} padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [16, 4] into %[[EMPTY]]
// CHECK-SAME:      : tensor<255x513xf32> -> tensor<16x129x16x4xf32>
// CHECK:         %[[EXPAND_LHS:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME:      output_shape [16, 129, 16, 1, 4, 1] : tensor<16x129x16x4xf32> into tensor<16x129x16x1x4x1xf32>
// CHECK:         %[[EMPTY_LHS2:.*]] = tensor.empty() : tensor<16x129x4x16x1x1xf32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SASME:     ins(%[[EXPAND_LHS]]
// CHECK-SAME:      outs(%[[EMPTY_LHS2]]
// CHECK-SAME:      permutation = [0, 1, 4, 2, 5, 3]
// CHECK:         %[[COLLAPSE:.*]] = tensor.collapse_shape %[[TRANSPOSE]]
// CHECK-SAME:      : tensor<16x129x4x16x1x1xf32> into tensor<16x129x64xf32>
// CHECK:         %[[EXPAND_LHS_2:.*]] = tensor.expand_shape %[[COLLAPSE]]
// CHECK-SAME:      : tensor<16x129x64xf32> into tensor<16x129x16x4xf32>
// CHECK:         flow.dispatch.tensor.store %[[EXPAND_LHS_2]]

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
// CHECK:         %[[EMPTY_RHS:.*]] = tensor.empty() : tensor<33x64x16x4xf32>
// CHECK:         %[[PACK_RHS:.*]] = tensor.pack %{{.+}} padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [1, 0]
// CHECK-SAME:      inner_dims_pos = [1, 0]
// CHECK-SAME:      inner_tiles = [16, 4] into %[[EMPTY_RHS]]
// CHECK-SAME:      : tensor<255x513xf32> -> tensor<33x64x16x4xf32>
// CHECK:         %[[EXPAND_RHS:.*]] = tensor.expand_shape %[[PACK_RHS]]
// CHECK-SAME:      output_shape [33, 64, 16, 1, 4, 1] : tensor<33x64x16x4xf32> into tensor<33x64x16x1x4x1xf32>
// CHECK:         %[[EMPTY_RHS2:.*]] = tensor.empty() : tensor<33x64x4x16x1x1xf32>
// CHECK:         %[[TRANSPOSE_RHS:.*]] = linalg.transpose
// CHECK-SAME:      ins(%[[EXPAND_RHS]]
// CHECK-SAME:      outs(%[[EMPTY_RHS2]]
// CHECK-SAME:      permutation = [0, 1, 4, 2, 5, 3]
// CHECK:         %[[COLLAPSE_RHS:.*]] = tensor.collapse_shape %[[TRANSPOSE_RHS]]
// CHECK-SAME:      : tensor<33x64x4x16x1x1xf32> into tensor<33x64x64xf32>
// CHECK:         %[[EXPAND_RHS_2:.*]] = tensor.expand_shape %[[COLLAPSE_RHS]]
// CHECK-SAME:      : tensor<33x64x64xf32> into tensor<33x64x16x4xf32>
// CHECK:         flow.dispatch.tensor.store %[[EXPAND_RHS_2]]

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
// CHECK:         %[[EMPTY_ACC:.*]] = tensor.empty() : tensor<16x33x16x16xf32>
// CHECK:         %[[PACK_ACC:.*]] = tensor.pack %{{.+}} padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [16, 16] into %[[EMPTY_ACC]]
// CHECK-SAME:      : tensor<255x513xf32> -> tensor<16x33x16x16xf32>
// CHECK:         %[[EXPAND_ACC:.*]] = tensor.expand_shape %[[PACK_ACC]]
// CHECK:         %[[EMPTY_ACC2:.*]] = tensor.empty() : tensor<16x33x4x16x4x1xf32>
// CHECK:         %[[TRANSPOSE_ACC:.*]] = linalg.transpose
// CHECK-SAME:      ins(%[[EXPAND_ACC]]
// CHECK-SAME:      outs(%[[EMPTY_ACC2]]
// CHECK-SAME:      permutation = [0, 1, 2, 4, 3, 5]
// CHECK:         %[[COLLAPSE_RHS:.*]] = tensor.collapse_shape %[[TRANSPOSE_ACC]]
// CHECK-SAME:      : tensor<16x33x4x16x4x1xf32> into tensor<16x33x256xf32>
// CHECK:         %[[EXPAND_ACC_2:.*]] = tensor.expand_shape %[[COLLAPSE_RHS]]
// CHECK-SAME:      : tensor<16x33x256xf32> into tensor<16x33x16x16xf32>
// CHECK:         flow.dispatch.tensor.store %[[EXPAND_ACC_2]]
