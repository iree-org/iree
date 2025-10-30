// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding{test-gpu-encoding-resolver=gpu_data_tiling}))" \
// RUN:   --iree-gpu-test-target=gfx942 \
// RUN:   --split-input-file %s | FileCheck %s

//-----------------------------------------------------------------------------
// 1. MFMA_F32_16x16x4_F32
//-----------------------------------------------------------------------------

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [255, 513, ?]>
func.func @empty_fill_encoding_unroll8x8x4_MFMA_F32_16x16x4_F32() -> tensor<255x513xf32, #encoding> {
  %0 = arith.constant 0.0 : f32
  %1 = tensor.empty() : tensor<255x513xf32, #encoding>
  %2 = linalg.fill ins(%0 : f32) outs(%1 : tensor<255x513xf32, #encoding>) -> tensor<255x513xf32, #encoding>
  return %2 : tensor<255x513xf32, #encoding>
}

// CHECK-LABEL: func.func @empty_fill_encoding_unroll8x8x4_MFMA_F32_16x16x4_F32
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<2x33x8x4x16x4xf32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins({{.+}}) outs(%[[EMPTY]]
// CHECK:         return %[[FILL]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [255, 513, ?]>
func.func @set_encoding_LHS_unroll8x8x4_MFMA_F32_16x16x4_F32(
    %arg0: tensor<255x513xf32>
) -> tensor<255x513xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  return %0 : tensor<255x513xf32, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_LHS_unroll8x8x4_MFMA_F32_16x16x4_F32
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[PACK:.*]] = linalg.pack %[[ARG0]] padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 16]
// CHECK-SAME:      : tensor<255x513xf32> -> tensor<2x33x128x16xf32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<2x33x128x16xf32> into tensor<2x33x4x8x4x4x4xf32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<2x33x8x16x4x4xf32>)
// CHECK-SAME:       outs({{.*}} : tensor<2x33x8x4x16x4xf32>)
// CHECK-SAME:       permutation = [0, 1, 2, 5, 3, 4]
// CHECK:         return %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [255, 513, ?]>
func.func @set_encoding_LHS_narrow_unroll1x8x4_MFMA_F32_16x16x4_F32(
    %arg0: tensor<255x513xf32>
) -> tensor<255x513xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  return %0 : tensor<255x513xf32, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_LHS_narrow_unroll1x8x4_MFMA_F32_16x16x4_F32
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[PACK:.*]] = linalg.pack %[[ARG0]] padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 16]
// CHECK-SAME:      : tensor<255x513xf32> -> tensor<2x33x128x16xf32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<2x33x128x16xf32> into tensor<2x33x4x8x4x4x4xf32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<2x33x8x16x4x4xf32>)
// CHECK-SAME:       outs({{.*}} : tensor<2x33x8x4x16x4xf32>)
// CHECK-SAME:       permutation = [0, 1, 2, 5, 3, 4]
// CHECK:         return %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [?, ?, ?]>
func.func @set_encoding_LHS_dynamic_unroll8x8x4_MFMA_F32_16x16x4_F32(
    %arg0: tensor<?x?xf32>
) -> tensor<?x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  return %0 : tensor<?x?xf32, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_LHS_dynamic_unroll8x8x4_MFMA_F32_16x16x4_F32
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[PACK:.*]] = linalg.pack %[[ARG0]] padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 16]
// CHECK-SAME:      : tensor<?x?xf32> -> tensor<?x?x128x16xf32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<?x?x128x16xf32> into tensor<?x?x4x8x4x4x4xf32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<?x?x8x16x4x4xf32>)
// CHECK-SAME:       outs({{.*}} : tensor<?x?x8x4x16x4xf32>)
// CHECK-SAME:       permutation = [0, 1, 2, 5, 3, 4]
// CHECK:         return %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [255, 513, ?]>
func.func @set_encoding_RHS_unroll8x8x4_MFMA_F32_16x16x4_F32(
    %arg0: tensor<255x513xf32>
) -> tensor<255x513xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  return %0 : tensor<255x513xf32, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_RHS_unroll8x8x4_MFMA_F32_16x16x4_F32
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[PACK:.*]] = linalg.pack %[[ARG0]] padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [1, 0]
// CHECK-SAME:      inner_dims_pos = [1, 0]
// CHECK-SAME:      inner_tiles = [128, 16]
// CHECK-SAME:      : tensor<255x513xf32> -> tensor<5x16x128x16xf32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<5x16x128x16xf32> into tensor<5x16x4x16x2x4x4xf32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<5x16x4x2x16x4x4xf32>)
// CHECK-SAME:       outs({{.*}} : tensor<5x16x4x2x4x16x4xf32>)
// CHECK-SAME:       permutation = [0, 1, 2, 3, 6, 4, 5]
// CHECK:         return %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [255, 513, ?]>
func.func @set_encoding_RHS_narrow_unroll8x1x4_MFMA_F32_16x16x4_F32(
    %arg0: tensor<255x513xf32>
) -> tensor<255x513xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  return %0 : tensor<255x513xf32, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_RHS_narrow_unroll8x1x4_MFMA_F32_16x16x4_F32
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[PACK:.*]] = linalg.pack %[[ARG0]] padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [1, 0]
// CHECK-SAME:      inner_dims_pos = [1, 0]
// CHECK-SAME:      inner_tiles = [128, 16]
// CHECK-SAME:      : tensor<255x513xf32> -> tensor<5x16x128x16xf32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<5x16x128x16xf32> into tensor<5x16x4x16x2x4x4xf32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<5x16x4x2x16x4x4xf32>)
// CHECK-SAME:       outs({{.*}} : tensor<5x16x4x2x4x16x4xf32>)
// CHECK-SAME:       permutation = [0, 1, 2, 3, 6, 4, 5]
// CHECK:         return %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [255, 513, ?]>
func.func @set_encoding_ACC_unroll8x8x4_MFMA_F32_16x16x4_F32(
    %arg0: tensor<255x513xf32>
) -> tensor<255x513xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  return %0 : tensor<255x513xf32, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_ACC_unroll8x8x4_MFMA_F32_16x16x4_F32
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[PACK:.*]] = linalg.pack %[[ARG0]] padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 128]
// CHECK-SAME:      : tensor<255x513xf32> -> tensor<2x5x128x128xf32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<2x5x128x128xf32> into tensor<2x5x4x8x4x4x16x2xf32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<2x5x8x4x4x4x2x16xf32>)
// CHECK-SAME:       outs({{.*}} : tensor<2x5x4x8x2x4x16x4xf32>)
// CHECK-SAME:       permutation = [0, 1, 5, 2, 6, 3, 7, 4]
// CHECK:         return %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [?, 513, ?]>
func.func @set_encoding_ACC_dynamic_M_MFMA_F32_16x16x4_F32(%arg0 : tensor<?x513xf32>) -> tensor<?x513xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x513xf32> -> tensor<?x513xf32, #encoding>
  return %0 : tensor<?x513xf32, #encoding>
}
// CHECK-LABEL: func.func @set_encoding_ACC_dynamic_M_MFMA_F32_16x16x4_F32
// CHECK:         %[[PACK:.*]] = linalg.pack %{{.+}} padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 128]
// CHECK-SAME:      : tensor<?x513xf32> -> tensor<?x5x128x128xf32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<?x5x128x128xf32> into tensor<?x5x4x4x2x4x16x8xf32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<?x5x4x2x4x4x8x16xf32>)
// CHECK-SAME:       outs({{.*}} : tensor<?x5x4x2x8x4x16x4xf32>)
// CHECK-SAME:       permutation = [0, 1, 2, 3, 6, 4, 7, 5]
// CHECK:         return %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [255, ?, ?]>
func.func @set_encoding_ACC_dynamic_N_MFMA_F32_16x16x4_F32(%arg0 : tensor<255x?xf32>) -> tensor<255x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<255x?xf32> -> tensor<255x?xf32, #encoding>
  return %0 : tensor<255x?xf32, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_ACC_dynamic_N_MFMA_F32_16x16x4_F32
// CHECK:         %[[PACK:.*]] = linalg.pack %{{.+}} padding_value(%{{.+}} : f32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 128]
// CHECK-SAME:      : tensor<255x?xf32> -> tensor<2x?x128x128xf32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<2x?x128x128xf32> into tensor<2x?x4x8x4x4x16x2xf32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<2x?x8x4x4x4x2x16xf32>)
// CHECK-SAME:       outs({{.*}} : tensor<2x?x4x8x2x4x16x4xf32>)
// CHECK-SAME:       permutation = [0, 1, 5, 2, 6, 3, 7, 4]
// CHECK:         return %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [255, 513, ?]>
func.func @unset_encoding_ACC_unroll8x8x4_MFMA_F32_16x16x4_F32(
    %arg0: tensor<255x513xf32, #encoding>
) -> tensor<255x513xf32> {
  %0 = iree_encoding.unset_encoding %arg0 : tensor<255x513xf32, #encoding> -> tensor<255x513xf32>
  return %0 : tensor<255x513xf32>
}

// CHECK-LABEL: func.func @unset_encoding_ACC_unroll8x8x4_MFMA_F32_16x16x4_F32
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[ARG0]] : tensor<2x5x4x8x2x4x16x4xf32>)
// CHECK-SAME:       outs({{.*}} : tensor<2x5x8x4x4x4x2x16xf32>)
// CHECK-SAME:       permutation = [0, 1, 3, 5, 7, 2, 4, 6]
// CHECK:         %[[COLLAPSE:.*]] = tensor.collapse_shape %[[TRANSPOSE]]
// CHECK-SAME:      : tensor<2x5x8x4x4x4x2x16xf32> into tensor<2x5x128x128xf32>
// CHECK:         %[[UNPACK:.*]] = linalg.unpack %[[COLLAPSE]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 128]
// CHECK-SAME:      : tensor<2x5x128x128xf32> -> tensor<255x513xf32>
// CHECK:         return %[[UNPACK]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [?, ?, ?]>
func.func @unset_encoding_ACC_dynamic_unroll8x8x4_MFMA_F32_16x16x4_F32(
    %arg0: tensor<?x?xf32, #encoding>, %d0: index, %d1: index
) -> tensor<?x?xf32> {
  %2 = iree_encoding.unset_encoding %arg0 : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>{%d0, %d1}
  return %2 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @unset_encoding_ACC_dynamic_unroll8x8x4_MFMA_F32_16x16x4_F32
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[ARG0]] : tensor<?x?x4x8x2x4x16x4xf32>)
// CHECK-SAME:       outs({{.*}} : tensor<?x?x8x4x4x4x2x16xf32>)
// CHECK-SAME:       permutation = [0, 1, 3, 5, 7, 2, 4, 6]
// CHECK:         %[[COLLAPSE:.*]] = tensor.collapse_shape %[[TRANSPOSE]]
// CHECK-SAME:      : tensor<?x?x8x4x4x4x2x16xf32> into tensor<?x?x128x128xf32>
// CHECK:         %[[UNPACK:.*]] = linalg.unpack %[[COLLAPSE]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 128]
// CHECK-SAME:      : tensor<?x?x128x128xf32> -> tensor<?x?xf32>
// CHECK:         return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_MFMA_F32_16x16x4_F32(
    %arg0: tensor<?x?xf32, #encoding_lhs>,
    %arg1: tensor<?x?xf32, #encoding_rhs>,
    %arg2: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result> {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xf32, #encoding_lhs>,
                         tensor<?x?xf32, #encoding_rhs>)
      outs(%arg2 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  return %0 : tensor<?x?xf32, #encoding_result>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK:     func.func @matmul_lowering_MFMA_F32_16x16x4_F32(
// CHECK-SAME:     %[[LHS:.+]]: tensor<?x?x8x4x16x4xf32>, %[[RHS:.+]]: tensor<?x?x4x2x4x16x4xf32>
// CHECK-SAME:     %[[ACC:.+]]: tensor<?x?x4x8x2x4x16x4xf32>
// CHECK-SAME:   ) -> tensor<?x?x4x8x2x4x16x4xf32>
// CHECK:       %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 4>
// CHECK:       return %[[MMA]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?, ?]>
func.func @batch_matmul_lowering_MFMA_F32_16x16x4_F32(
    %arg0: tensor<?x?x?xf32, #encoding_lhs>,
    %arg1: tensor<?x?x?xf32, #encoding_rhs>,
    %arg2: tensor<?x?x?xf32, #encoding_result>
) -> tensor<?x?x?xf32, #encoding_result> {
  %0 = linalg.batch_matmul
      ins(%arg0, %arg1 : tensor<?x?x?xf32, #encoding_lhs>,
                         tensor<?x?x?xf32, #encoding_rhs>)
      outs(%arg2 : tensor<?x?x?xf32, #encoding_result>)
      -> tensor<?x?x?xf32, #encoding_result>
  return %0 : tensor<?x?x?xf32, #encoding_result>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK:     func.func @batch_matmul_lowering_MFMA_F32_16x16x4_F32(
// CHECK-SAME:     %[[LHS:.+]]: tensor<?x?x?x8x4x16x4xf32>
// CHECK-SAME:     %[[RHS:.+]]: tensor<?x?x?x4x2x4x16x4xf32>
// CHECK-SAME:     %[[ACC:.+]]: tensor<?x?x?x4x8x2x4x16x4xf32>
// CHECK-SAME:   ) -> tensor<?x?x?x4x8x2x4x16x4xf32>
// CHECK:       %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x4_F32, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 4>
// CHECK:       return %[[MMA]]

// -----

//-----------------------------------------------------------------------------
// 2. MFMA_I32_16x16x32_I8
//-----------------------------------------------------------------------------

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [255, 513, ?]>
func.func @set_encoding_LHS_unroll8x8x2_MFMA_I32_16x16x32_I8(
    %arg0: tensor<255x513xi8>
) -> tensor<255x513xi8, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<255x513xi8> -> tensor<255x513xi8, #encoding>
  return %0 : tensor<255x513xi8, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_LHS_unroll8x8x2_MFMA_I32_16x16x32_I8
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[PACK:.*]] = linalg.pack %[[ARG0]] padding_value(%{{.+}} : i8)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 64]
// CHECK-SAME:      : tensor<255x513xi8> -> tensor<2x9x128x64xi8>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<2x9x128x64xi8> into tensor<2x9x4x8x4x2x4x8xi8>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<2x9x8x16x2x4x8xi8>)
// CHECK-SAME:       outs({{.*}} : tensor<2x9x8x4x16x2x8xi8>)
// CHECK-SAME:       permutation = [0, 1, 2, 5, 3, 4, 6]
// CHECK:         return %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [255, 513, ?]>
func.func @set_encoding_RHS_unroll8x8x2_MFMA_I32_16x16x32_I8(
    %arg0: tensor<255x513xi8>
) -> tensor<255x513xi8, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<255x513xi8> -> tensor<255x513xi8, #encoding>
  return %0 : tensor<255x513xi8, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_RHS_unroll8x8x2_MFMA_I32_16x16x32_I8
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[PACK:.*]] = linalg.pack %[[ARG0]] padding_value(%{{.+}} : i8)
// CHECK-SAME:      outer_dims_perm = [1, 0]
// CHECK-SAME:      inner_dims_pos = [1, 0]
// CHECK-SAME:      inner_tiles = [128, 64]
// CHECK-SAME:      : tensor<255x513xi8> -> tensor<5x4x128x64xi8>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<5x4x128x64xi8> into tensor<5x4x4x16x2x2x4x8xi8>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<5x4x4x2x16x2x4x8xi8>)
// CHECK-SAME:       outs({{.*}} : tensor<5x4x4x2x4x16x2x8xi8>)
// CHECK-SAME:       permutation = [0, 1, 2, 3, 6, 4, 5, 7]
// CHECK:         return %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [255, 513, ?]>
func.func @set_encoding_ACC_unroll8x8x2_MFMA_I32_16x16x32_I8(
    %arg0: tensor<255x513xi32>
) -> tensor<255x513xi32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<255x513xi32> -> tensor<255x513xi32, #encoding>
  return %0 : tensor<255x513xi32, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_ACC_unroll8x8x2_MFMA_I32_16x16x32_I8
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[PACK:.*]] = linalg.pack %[[ARG0]] padding_value(%{{.+}} : i32)
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 128]
// CHECK-SAME:      : tensor<255x513xi32> -> tensor<2x5x128x128xi32>
// CHECK:         %[[EXPAND:.*]] = tensor.expand_shape %[[PACK]]
// CHECK-SAME       : tensor<2x5x128x128xi32> into tensor<2x5x4x8x4x4x16x2xi32>
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[EXPAND]] : tensor<2x5x8x4x4x4x2x16xi32>)
// CHECK-SAME:       outs({{.*}} : tensor<2x5x4x8x2x4x16x4xi32>)
// CHECK-SAME:       permutation = [0, 1, 5, 2, 6, 3, 7, 4]
// CHECK:         return %[[TRANSPOSE]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
                                    iteration_sizes = [255, 513, ?]>
func.func @unset_encoding_ACC_unroll8x8x2_MFMA_I32_16x16x32_I8(
    %arg0: tensor<255x513xi32, #encoding>
) -> tensor<255x513xi32> {
  %0 = iree_encoding.unset_encoding %arg0 : tensor<255x513xi32, #encoding> -> tensor<255x513xi32>
  return %0 : tensor<255x513xi32>
}

// CHECK-LABEL: func.func @unset_encoding_ACC_unroll8x8x2_MFMA_I32_16x16x32_I8
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[TRANSPOSE:.*]] = linalg.transpose
// CHECK-SAME:       ins(%[[ARG0]] : tensor<2x5x4x8x2x4x16x4xi32>)
// CHECK-SAME:       outs({{.*}} : tensor<2x5x8x4x4x4x2x16xi32>)
// CHECK-SAME:       permutation = [0, 1, 3, 5, 7, 2, 4, 6]
// CHECK:         %[[COLLAPSE:.*]] = tensor.collapse_shape %[[TRANSPOSE]]
// CHECK-SAME:      : tensor<2x5x8x4x4x4x2x16xi32> into tensor<2x5x128x128xi32>
// CHECK:         %[[UNPACK:.*]] = linalg.unpack %[[COLLAPSE]]
// CHECK-SAME:      outer_dims_perm = [0, 1]
// CHECK-SAME:      inner_dims_pos = [0, 1]
// CHECK-SAME:      inner_tiles = [128, 128]
// CHECK-SAME:      : tensor<2x5x128x128xi32> -> tensor<255x513xi32>
// CHECK:         return %[[UNPACK]]

// -----

#encoding = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f16, f16, f32],
                                    user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1) -> ()>]]>
func.func @set_encoding_0D_tensor(
    %arg0: tensor<f32>
) -> tensor<f32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<f32> -> tensor<f32, #encoding>
  return %0 : tensor<f32, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_0D_tensor
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9]+]]: tensor<f32>
// CHECK:         return %[[ARG0]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_MFMA_I32_16x16x32_I8(
    %arg0: tensor<?x?xi8, #encoding_lhs>,
    %arg1: tensor<?x?xi8, #encoding_rhs>,
    %arg2: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xi8, #encoding_lhs>,
                       tensor<?x?xi8, #encoding_rhs>)
      outs(%arg2 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %0 : tensor<?x?xi32, #encoding_result>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK:     func.func @matmul_lowering_MFMA_I32_16x16x32_I8(
// CHECK-SAME:     %[[LHS:.+]]: tensor<?x?x8x4x16x2x8xi8>
// CHECK-SAME:     %[[RHS:.+]]: tensor<?x?x4x2x4x16x2x8xi8>
// CHECK-SAME:     %[[ACC:.+]]: tensor<?x?x4x8x2x4x16x4xi32>
// CHECK-SAME:   ) -> tensor<?x?x4x8x2x4x16x4xi32>
// CHECK:       %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x32_I8, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 2>
// CHECK:       return %[[MMA]]

// -----

//-------------------------------------------------------------------------
// 3. Custom target parameters to test more MaterializeEncoding heuristics.
//-------------------------------------------------------------------------

// Custom {max_load_instruction_bits = 64} => implied default {intrinsics_k = 1} (omitted in output) instead of {intrinsics_k = 2}.

#target_gfx942_except_max_load_instruction_bits_64 = #hal.executable.target<"rocm", "rocm-hsaco-fb", {
  iree_codegen.target_info = #iree_gpu.target<
    arch = "gfx942", features = "", wgp = <
      compute =  fp64|fp32|fp16|int64|int32|int16|int8,
      storage =  b64|b32|b16|b8,
      subgroup = shuffle|arithmetic,
      dot =  dp4xi8toi32,
      mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>],
      subgroup_size_choices = [64],
      max_workgroup_sizes = [1024, 1024, 1024],
      max_thread_count_per_workgroup = 1024,
      max_workgroup_memory_bytes = 65536,
      max_workgroup_counts = [2147483647, 2147483647, 2147483647],
      max_load_instruction_bits = 64,
      simds_per_wgp = 4,
      vgpr_space_bits = 16384
    >
  >,
  ukernels = "none",
  iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>
}>

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_MFMA_I32_16x16x32_I8_custom_max_load_instruction_bits_64(
    %arg0: tensor<?x?xi8, #encoding_lhs>,
    %arg1: tensor<?x?xi8, #encoding_rhs>,
    %arg2: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {hal.executable.target = #target_gfx942_except_max_load_instruction_bits_64} {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xi8, #encoding_lhs>,
                         tensor<?x?xi8, #encoding_rhs>)
      outs(%arg2 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %0 : tensor<?x?xi32, #encoding_result>
}

// CHECK:      func.func @matmul_lowering_MFMA_I32_16x16x32_I8_custom_max_load_instruction_bits_64
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ACC:[a-zA-Z0-9]+]]
// CHECK:      iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:     kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x32_I8, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4>

// -----

// Custom {max_load_instruction_bits = 256} => {intrinsics_k = 4} instead of {intrinsics_k = 2}.

#target_gfx942_except_max_load_instruction_bits_256 = #hal.executable.target<"rocm", "rocm-hsaco-fb", {
  iree_codegen.target_info = #iree_gpu.target<
    arch = "gfx942", features = "", wgp = <
      compute =  fp64|fp32|fp16|int64|int32|int16|int8,
      storage =  b64|b32|b16|b8,
      subgroup = shuffle|arithmetic,
      dot =  dp4xi8toi32,
      mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>],
      subgroup_size_choices = [64],
      max_workgroup_sizes = [1024, 1024, 1024],
      max_thread_count_per_workgroup = 1024,
      max_workgroup_memory_bytes = 65536,
      max_workgroup_counts = [2147483647, 2147483647, 2147483647],
      max_load_instruction_bits = 256,
      simds_per_wgp = 4,
      vgpr_space_bits = 16384
    >
  >,
  ukernels = "none",
  iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>
}>

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_MFMA_I32_16x16x32_I8_custom_max_load_instruction_bits_64(
    %arg0: tensor<?x?xi8, #encoding_lhs>,
    %arg1: tensor<?x?xi8, #encoding_rhs>,
    %arg2: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {hal.executable.target = #target_gfx942_except_max_load_instruction_bits_256} {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xi8, #encoding_lhs>,
                         tensor<?x?xi8, #encoding_rhs>)
      outs(%arg2 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %0 : tensor<?x?xi32, #encoding_result>
}

// CHECK:      func.func @matmul_lowering_MFMA_I32_16x16x32_I8_custom_max_load_instruction_bits_64
// CHECK-SAME:     %[[LHS:[a-zA-Z0-9_]+]]
// CHECK-SAME:     %[[RHS:[a-zA-Z0-9_]+]]
// CHECK-SAME:     %[[ACC:[a-zA-Z0-9_]+]]
// CHECK:      iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:     kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x32_I8, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 4>

// -----

// Custom {simds_per_wgp = 1} => implied default {subgroups_n = 1} (omitted in output) and {intrinsics_n = 8} instead of {subgroups_n = 4}.

#target_gfx942_except_simds_per_wgp_1 = #hal.executable.target<"rocm", "rocm-hsaco-fb", {
  iree_codegen.target_info = #iree_gpu.target<
    arch = "gfx942", features = "", wgp = <
      compute =  fp64|fp32|fp16|int64|int32|int16|int8,
      storage =  b64|b32|b16|b8,
      subgroup = shuffle|arithmetic,
      dot =  dp4xi8toi32,
      mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>],
      subgroup_size_choices = [64],
      max_workgroup_sizes = [1024, 1024, 1024],
      max_thread_count_per_workgroup = 1024,
      max_workgroup_memory_bytes = 65536,
      max_workgroup_counts = [2147483647, 2147483647, 2147483647],
      max_load_instruction_bits = 128,
      simds_per_wgp = 1,
      vgpr_space_bits = 16384
    >
  >,
  ukernels = "none",
  iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>
}>

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_MFMA_I32_16x16x32_I8_custom_simds_per_wgp_1(
    %arg0: tensor<?x?xi8, #encoding_lhs>,
    %arg1: tensor<?x?xi8, #encoding_rhs>,
    %arg2: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {hal.executable.target = #target_gfx942_except_simds_per_wgp_1} {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xi8, #encoding_lhs>,
                         tensor<?x?xi8, #encoding_rhs>)
      outs(%arg2 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %0 : tensor<?x?xi32, #encoding_result>
}

// CHECK:      func.func @matmul_lowering_MFMA_I32_16x16x32_I8_custom_simds_per_wgp_1
// CHECK-SAME:     %[[LHS:[a-zA-Z0-9_]+]]
// CHECK-SAME:     %[[RHS:[a-zA-Z0-9_]+]]
// CHECK-SAME:     %[[ACC:[a-zA-Z0-9_]+]]
// CHECK:      iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:     kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x32_I8, intrinsics_m = 8, intrinsics_n = 8, intrinsics_k = 2>

// -----

// Custom 2x smaller {vgpr_space_bits = 8192} => smaller intrinsics_m and intrinsics_n

#target_gfx942_except_vgpr_space_bits_8192 = #hal.executable.target<"rocm", "rocm-hsaco-fb", {
  iree_codegen.target_info = #iree_gpu.target<
    arch = "gfx942", features = "", wgp = <
      compute =  fp64|fp32|fp16|int64|int32|int16|int8,
      storage =  b64|b32|b16|b8,
      subgroup = shuffle|arithmetic,
      dot =  dp4xi8toi32,
      mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>],
      subgroup_size_choices = [64],
      max_workgroup_sizes = [1024, 1024, 1024],
      max_thread_count_per_workgroup = 1024,
      max_workgroup_memory_bytes = 65536,
      max_workgroup_counts = [2147483647, 2147483647, 2147483647],
      max_load_instruction_bits = 128,
      simds_per_wgp = 4,
      vgpr_space_bits = 8192
    >
  >,
  ukernels = "none",
  iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>
}>

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_MFMA_I32_16x16x32_I8_custom_vgpr_space_bits_8192(
    %arg0: tensor<?x?xi8, #encoding_lhs>,
    %arg1: tensor<?x?xi8, #encoding_rhs>,
    %arg2: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {hal.executable.target = #target_gfx942_except_vgpr_space_bits_8192} {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xi8, #encoding_lhs>,
                         tensor<?x?xi8, #encoding_rhs>)
      outs(%arg2 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %0 : tensor<?x?xi32, #encoding_result>
}

// CHECK:      func.func @matmul_lowering_MFMA_I32_16x16x32_I8_custom_vgpr_space_bits_8192
// CHECK-SAME:     %[[LHS:[a-zA-Z0-9_]+]]
// CHECK-SAME:     %[[RHS:[a-zA-Z0-9_]+]]
// CHECK-SAME:     %[[ACC:[a-zA-Z0-9_]+]]
// CHECK:      iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:     kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x32_I8, intrinsics_m = 4, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 2>

// -----

// Custom 4x smaller {vgpr_space_bits = 4096} => smaller intrinsics_m and intrinsics_n

#target_gfx942_except_vgpr_space_bits_4096 = #hal.executable.target<"rocm", "rocm-hsaco-fb", {
  iree_codegen.target_info = #iree_gpu.target<
    arch = "gfx942", features = "", wgp = <
      compute =  fp64|fp32|fp16|int64|int32|int16|int8,
      storage =  b64|b32|b16|b8,
      subgroup = shuffle|arithmetic,
      dot =  dp4xi8toi32,
      mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>],
      subgroup_size_choices = [64],
      max_workgroup_sizes = [1024, 1024, 1024],
      max_thread_count_per_workgroup = 1024,
      max_workgroup_memory_bytes = 65536,
      max_workgroup_counts = [2147483647, 2147483647, 2147483647],
      max_load_instruction_bits = 128,
      simds_per_wgp = 4,
      vgpr_space_bits = 4096
    >
  >,
  ukernels = "none",
  iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>
}>

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_MFMA_I32_16x16x32_I8_custom_vgpr_space_bits_4096(
    %arg0: tensor<?x?xi8, #encoding_lhs>,
    %arg1: tensor<?x?xi8, #encoding_rhs>,
    %arg2: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {hal.executable.target = #target_gfx942_except_vgpr_space_bits_4096} {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xi8, #encoding_lhs>,
                         tensor<?x?xi8, #encoding_rhs>)
      outs(%arg2 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %0 : tensor<?x?xi32, #encoding_result>
}

// CHECK:      func.func @matmul_lowering_MFMA_I32_16x16x32_I8_custom_vgpr_space_bits_4096
// CHECK-SAME:     %[[LHS:.+]]: tensor<?x?x4x4x16x2x8xi8>, %[[RHS:.+]]: tensor<?x?x4x4x16x2x8xi8>
// CHECK-SAME:     %[[ACC:.+]]: tensor<?x?x4x4x4x16x4xi32>
// CHECK:      iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:     kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x32_I8,  intrinsics_m = 4, subgroups_n = 4, intrinsics_k = 2>

// -----

// Custom smaller {vgpr_space_bits = 32768} => larger intrinsics_m and/or intrinsics_n

#target_gfx942_except_vgpr_space_bits_32768 = #hal.executable.target<"rocm", "rocm-hsaco-fb", {
  iree_codegen.target_info = #iree_gpu.target<
    arch = "gfx942", features = "", wgp = <
      compute =  fp64|fp32|fp16|int64|int32|int16|int8,
      storage =  b64|b32|b16|b8,
      subgroup = shuffle|arithmetic,
      dot =  dp4xi8toi32,
      mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>, <MFMA_F32_32x32x8_F16>, <MFMA_F32_16x16x32_F8E4M3FNUZ>, <MFMA_I32_16x16x32_I8>, <MFMA_I32_32x32x16_I8>],
      subgroup_size_choices = [64],
      max_workgroup_sizes = [1024, 1024, 1024],
      max_thread_count_per_workgroup = 1024,
      max_workgroup_memory_bytes = 65536,
      max_workgroup_counts = [2147483647, 2147483647, 2147483647],
      max_load_instruction_bits = 128,
      simds_per_wgp = 4,
      vgpr_space_bits = 32768
    >
  >,
  ukernels = "none",
  iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>
}>

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_MFMA_I32_16x16x32_I8_custom_vgpr_space_bits_32768(
    %arg0: tensor<?x?xi8, #encoding_lhs>,
    %arg1: tensor<?x?xi8, #encoding_rhs>,
    %arg2: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {hal.executable.target = #target_gfx942_except_vgpr_space_bits_32768} {
  %0 = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x?xi8, #encoding_lhs>,
                         tensor<?x?xi8, #encoding_rhs>)
      outs(%arg2 : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %0 : tensor<?x?xi32, #encoding_result>
}

// CHECK:      func.func @matmul_lowering_MFMA_I32_16x16x32_I8_custom_vgpr_space_bits_32768
// CHECK-SAME:     %[[LHS:.+]]:  tensor<?x?x8x4x16x2x8xi8>
// CHECK-SAME:     %[[RHS:.+]]: tensor<?x?x4x4x4x16x2x8xi8>
// CHECK-SAME:     %[[ACC:.+]]: tensor<?x?x4x8x4x4x16x4xi32>
// CHECK:      iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:     iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:     kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x32_I8, intrinsics_m = 8, intrinsics_n = 4, subgroups_n = 4, intrinsics_k = 2>

// -----

//---------------------------------------------------------------------------
// 4. Additional element types, testing only the multi_mma, not set_encoding.
//---------------------------------------------------------------------------

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?, ?]>
func.func @batch_matmul_lowering_MFMA_F32_16x16x32_F8E4M3FNUZ(
    %arg0: tensor<?x?x?xf8E4M3FNUZ, #encoding_lhs>,
    %arg1: tensor<?x?x?xf8E4M3FNUZ, #encoding_rhs>,
    %arg2: tensor<?x?x?xf32, #encoding_result>
) -> tensor<?x?x?xf32, #encoding_result> {
  %0 = linalg.batch_matmul
      ins(%arg0, %arg1 : tensor<?x?x?xf8E4M3FNUZ, #encoding_lhs>,
                         tensor<?x?x?xf8E4M3FNUZ, #encoding_rhs>)
      outs(%arg2 : tensor<?x?x?xf32, #encoding_result>)
      -> tensor<?x?x?xf32, #encoding_result>
  return %0 : tensor<?x?x?xf32, #encoding_result>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK:     func.func @batch_matmul_lowering_MFMA_F32_16x16x32_F8E4M3FNUZ(
// CHECK-SAME:     %[[LHS:[a-zA-Z0-9_]+]]: tensor<?x?x?x8x4x16x2x8xf8E4M3FNUZ>
// CHECK-SAME:     %[[RHS:[a-zA-Z0-9_]+]]: tensor<?x?x?x4x2x4x16x2x8xf8E4M3FNUZ>
// CHECK-SAME:     %[[ACC:[a-zA-Z0-9_]+]]: tensor<?x?x?x4x8x2x4x16x4xf32>
// CHECK-SAME:   ) -> tensor<?x?x?x4x8x2x4x16x4xf32>
// CHECK:       %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x32_F8E4M3FNUZ, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 2>
// CHECK:       return %[[MMA]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2]>
func.func @batch_matmul_lowering_MFMA_F32_16x16x16_BF16(
    %arg0: tensor<?x?x?xbf16, #encoding_lhs>,
    %arg1: tensor<?x?x?xbf16, #encoding_rhs>,
    %arg2: tensor<?x?x?xf32, #encoding_result>
) -> tensor<?x?x?xf32, #encoding_result> {
  %0 = linalg.batch_matmul
      ins(%arg0, %arg1 : tensor<?x?x?xbf16, #encoding_lhs>,
                         tensor<?x?x?xbf16, #encoding_rhs>)
      outs(%arg2 : tensor<?x?x?xf32, #encoding_result>)
      -> tensor<?x?x?xf32, #encoding_result>
  return %0 : tensor<?x?x?xf32, #encoding_result>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK:     func.func @batch_matmul_lowering_MFMA_F32_16x16x16_BF16(
// CHECK-SAME:     %[[LHS:.+]]: tensor<?x?x?x8x4x16x2x4xbf16>
// CHECK-SAME:     %[[RHS:.+]]: tensor<?x?x?x4x2x4x16x2x4xbf16>
// CHECK-SAME:     %[[ACC:.+]]: tensor<?x?x?x4x8x2x4x16x4xf32>
// CHECK-SAME:   ) -> tensor<?x?x?x4x8x2x4x16x4xf32>
// CHECK:       %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x16_BF16, intrinsics_m = 8, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 2>
// CHECK:       return %[[MMA]]

// -----

//----------------------------------------------------------------------------//
// Test suite for linalg.generic ops.
//----------------------------------------------------------------------------//

#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]>
#encoding_bcast = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [[affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2) -> (d0, d2)>], affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]>
func.func @dequantization(
    %arg0: tensor<2x128x64xi8, #encoding>,
    %arg1: tensor<2x64xf32, #encoding_bcast>,
    %arg2: tensor<2x64xf32, #encoding_bcast>
) -> tensor<2x128x64xf32, #encoding> {
  %13 = tensor.empty() : tensor<2x128x64xf32, #encoding>
  %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<2x128x64xi8, #encoding>, tensor<2x64xf32, #encoding_bcast>, tensor<2x64xf32, #encoding_bcast>) outs(%13 : tensor<2x128x64xf32, #encoding>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %21 = arith.extui %in : i8 to i32
      %22 = arith.uitofp %21 : i32 to f32
      %23 = arith.subf %22, %in_1 : f32
      %24 = arith.mulf %23, %in_0 : f32
      linalg.yield %24 : f32
  } -> tensor<2x128x64xf32, #encoding>
  return %14 : tensor<2x128x64xf32, #encoding>
}

// CHECK-DAG:   #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
// CHECK-DAG:   #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d4, d6)>
// CHECK-LABEL: func.func @dequantization
// CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<2x1x4x8x4x16x4xi8>
// CHECK-SAME:     %[[SCALES:[a-zA-Z0-9]+]]: tensor<2x4x4x4xf32>
// CHECK-SAME:     %[[ZPS:[a-zA-Z0-9]+]]: tensor<2x4x4x4xf32>
// CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<2x1x4x8x4x16x4xf32>
// CHECK:         %[[DEQUANT:.+]] = linalg.generic
// CHECK-SAME:        indexing_maps = [#[[$MAP]], #[[$MAP1]], #[[$MAP1]], #[[$MAP]]]
// CHECK-SAME:        iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:        ins(%[[LHS]], %[[SCALES]], %[[ZPS]]
// CHECK-SAME:        outs(%[[EMPTY]]
// CHECK:           arith.extui
// CHECK:           arith.uitofp
// CHECK:           arith.subf
// CHECK:           arith.mulf
// CHECK:         return %[[DEQUANT]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]>
func.func @multi_result_generic(%3: tensor<2x128x64xf32, #encoding>) -> (tensor<2x128x64xf32, #encoding>, tensor<2x128x64xf16, #encoding>) {
  %4 = tensor.empty() : tensor<2x128x64xf32, #encoding>
  %5 = tensor.empty() : tensor<2x128x64xf16, #encoding>
  %15:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%3 : tensor<2x128x64xf32, #encoding>)
      outs(%4, %5 : tensor<2x128x64xf32, #encoding>, tensor<2x128x64xf16, #encoding>) {
  ^bb0(%in: f32, %out: f32, %out_0: f16):
    %6 = arith.addf %in, %in : f32
    %7 = arith.truncf %6 : f32 to f16
    linalg.yield %6, %7 : f32, f16
  } -> (tensor<2x128x64xf32, #encoding>, tensor<2x128x64xf16, #encoding>)
  return %15#0, %15#1 : tensor<2x128x64xf32, #encoding>, tensor<2x128x64xf16, #encoding>
}
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
// CHECK-LABEL: func.func @multi_result_generic(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x1x4x8x4x16x4xf32>
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP]], #[[$MAP]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG0]]
//       CHECK:     arith.addf
//       CHECK:     arith.truncf
//       CHECK:     -> (tensor<2x1x4x8x4x16x4xf32>, tensor<2x1x4x8x4x16x4xf16>)

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]>
#output_encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [[affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2) -> (d2, d0, d1)>], affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]>
func.func @interchange_generic(%3: tensor<2x128x64xf32, #encoding>) -> tensor<64x2x128xf32, #output_encoding> {
  %4 = tensor.empty() : tensor<64x2x128xf32, #output_encoding>
  %15 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%3 : tensor<2x128x64xf32, #encoding>)
      outs(%4 : tensor<64x2x128xf32, #output_encoding>) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %in : f32
    linalg.yield %6 : f32
  } -> tensor<64x2x128xf32, #output_encoding>
  return %15 : tensor<64x2x128xf32, #output_encoding>
}
//   CHECK-DAG: #[[$MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
// CHECK-LABEL: func.func @interchange_generic(
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<2x1x4x8x4x16x4xf32>
//       CHECK:   linalg.generic
//  CHECK-SAME:       indexing_maps = [#[[$MAP]], #[[$MAP]]]
//  CHECK-SAME:       iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]
//  CHECK-SAME:       ins(%[[ARG0]]
//       CHECK:     arith.addf
//       CHECK:     -> tensor<2x1x4x8x4x16x4xf32>
// -----

//----------------------------------------------------------------------------//
// Test suite for encodings with resolved layouts.
//----------------------------------------------------------------------------//

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>}>
#encoding = #iree_encoding.layout<[#iree_gpu.gpu_encoding_resolver<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [128, 16], outerDimsPerm = [0, 1]}}>]>
func.func @set_encoding_with_layout(
    %arg0: tensor<255x513xf32>
) -> tensor<255x513xf32, #encoding> attributes {
  hal.executable.target = #executable_target_rocm_hsaco_fb
} {
  %0 = iree_encoding.set_encoding %arg0 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  return %0 : tensor<255x513xf32, #encoding>
}

// CHECK-LABEL: func.func @set_encoding_with_layout
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         %[[PACK:.*]] = linalg.pack %[[ARG0]]
// CHECK-SAME:     outer_dims_perm = [0, 1]
// CHECK-SAME:     inner_dims_pos = [0, 1]
// CHECK-SAME:     inner_tiles = [128, 16]
// CHECK-SAME:     tensor<255x513xf32> -> tensor<2x33x128x16xf32>
// CHECK:         return %[[PACK]]

// -----

//------------------------------------------------------------------------------
// Negative tests. The pass should do nothing for these cases.
//------------------------------------------------------------------------------

// This test ensures that no side-effects happen/errors are thrown in case of
// missing encoding information like indexing maps.

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @missing_user_indexing_maps() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<255x513xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<255x513xf32, #encoding>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<255x513xf32>> -> tensor<255x513xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<255x513xf32> -> tensor<255x513xf32, #encoding>
  iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [255, 513], strides = [1, 1] : tensor<255x513xf32, #encoding> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<255x513xf32,  #encoding>>
  return
}

// CHECK-LABEL: func.func @missing_user_indexing_maps
// CHECK-DAG:     %[[LOAD_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(0)
// CHECK-DAG:     %[[STORE_BINDING:.+]] = hal.interface.binding.subspan {{.+}} binding(1)
// CHECK-DAG:     %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LOAD_BINDING]]{{.+}} -> tensor<255x513xf32>
// CHECK-DAG:     iree_tensor_ext.dispatch.tensor.store %[[LOAD]], %[[STORE_BINDING]]
