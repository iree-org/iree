// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding),canonicalize,cse)" --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
func.func @matmul_lowering_i8i8i32_vmvx_ukernel() attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "all"}>
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xi8, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
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
//  CHECK-DAG: #[[MAP_CEILDIV:.+]] = affine_map<()[s0, s1] -> (s0 ceildiv s1)>
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: #[[LHS_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>
//      CHECK: #[[RHS_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 1 : i64, op_type =  matmul, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>
//      CHECK: #[[OUT_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 2 : i64, op_type =  matmul, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>
//      CHECK: func @matmul_lowering_i8i8i32_vmvx_ukernel()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//      CHECK:   %[[LHS_TILE_SIZES:.+]]:2 = iree_codegen.query_tile_sizes tensor<?x?xi8, #[[LHS_ENCODING]]> -> index, index
//  CHECK-DAG:   %[[LHS_OUTER_SIZE0:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[M]], %[[LHS_TILE_SIZES]]#0]
//  CHECK-DAG:   %[[LHS_OUTER_SIZE1:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[K]], %[[LHS_TILE_SIZES]]#1]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%[[LHS_OUTER_SIZE0]], %[[LHS_OUTER_SIZE1]], %[[LHS_TILE_SIZES]]#0, %[[LHS_TILE_SIZES]]#1}
//      CHECK:   %[[RHS_TILE_SIZES:.+]]:2 = iree_codegen.query_tile_sizes tensor<?x?xi8, #[[RHS_ENCODING]]> -> index, index
//  CHECK-DAG:   %[[RHS_OUTER_SIZE0:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[N]], %[[RHS_TILE_SIZES]]#0]
//  CHECK-DAG:   %[[RHS_OUTER_SIZE1:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[K]], %[[RHS_TILE_SIZES]]#1]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x?x?xi8>>{%[[RHS_OUTER_SIZE0]], %[[RHS_OUTER_SIZE1]], %[[RHS_TILE_SIZES]]#0, %[[RHS_TILE_SIZES]]#1}
//      CHECK:   %[[RESULT_TILE_SIZES:.+]]:2 = iree_codegen.query_tile_sizes tensor<?x?xi32, #[[OUT_ENCODING]]> -> index, index
//  CHECK-DAG:   %[[RESULT_OUTER_SIZE0:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[M]], %[[RESULT_TILE_SIZES]]#0]
//  CHECK-DAG:   %[[RESULT_OUTER_SIZE1:.+]] = affine.apply #[[MAP_CEILDIV]]()[%[[N]], %[[RESULT_TILE_SIZES]]#1]
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x?x?xi32>>{%[[RESULT_OUTER_SIZE0]], %[[RESULT_OUTER_SIZE1]], %[[RESULT_TILE_SIZES]]#0, %[[RESULT_TILE_SIZES]]#1}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[LHS_OUTER_SIZE0]], %[[LHS_OUTER_SIZE1]], %[[LHS_TILE_SIZES]]#0, %[[LHS_TILE_SIZES]]#1], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[RHS_OUTER_SIZE0]], %[[RHS_OUTER_SIZE1]], %[[RHS_TILE_SIZES]]#0, %[[RHS_TILE_SIZES]]#1], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[RESULT_OUTER_SIZE0]], %[[RESULT_OUTER_SIZE1]], %[[RESULT_TILE_SIZES]]#0, %[[RESULT_TILE_SIZES]]#1], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[RESULT_OUTER_SIZE0]], %[[RESULT_OUTER_SIZE1]], %[[RESULT_TILE_SIZES]]#0, %[[RESULT_TILE_SIZES]]#1], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<()[s0] -> ((3 ceildiv s0) * s0)>
#map1 = affine_map<()[s0] -> ((1 ceildiv s0) * s0)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map2, #map3, #map4], round_dims_to = array<i64: 32, 32, 32>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map2, #map3, #map4], round_dims_to = array<i64: 32, 32, 32>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map2, #map3, #map4], round_dims_to = array<i64: 32, 32, 32>>
func.func @fill_matmul(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index) attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
} {
  %c32_i64 = arith.constant 32 : i64
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x2xf32, #encoding_lhs>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x3xf32, #encoding_rhs>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x3xf32, #encoding_result>>{%arg4, %arg5}
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x2xf32, #encoding_lhs>> -> tensor<1x2xf32, #encoding_lhs>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2, 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x3xf32, #encoding_rhs>> -> tensor<2x3xf32, #encoding_rhs>
  %7 = tensor.empty() : tensor<1x3xf32, #encoding_result>
  %8 = linalg.fill ins(%cst : f32) outs(%7 : tensor<1x3xf32, #encoding_result>) -> tensor<1x3xf32, #encoding_result>
  %9 = linalg.matmul ins(%3, %4 : tensor<1x2xf32, #encoding_lhs>, tensor<2x3xf32, #encoding_rhs>) outs(%8 : tensor<1x3xf32, #encoding_result>) -> tensor<1x3xf32, #encoding_result>
  flow.dispatch.tensor.store %9, %2, offsets = [0, 0], sizes = [1, 3], strides = [1, 1] : tensor<1x3xf32, #encoding_result> -> !flow.dispatch.tensor<writeonly:tensor<1x3xf32, #encoding_result>>
  return
}
//      CHECK: func.func @fill_matmul
//  CHECK-DAG:   %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<1x1x8x4xf32>>
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<1x1x8x4xf32>>
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<writeonly:tensor<1x1x8x8xf32>>
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<1x1x8x8xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:     ins(%[[ZERO]]
// CHECK-SAME:     outs(%[[EMPTY]]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:     ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:     outs(%[[FILL]]
//      CHECK:  flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:    offsets = [0, 0, 0, 0], sizes = [1, 1, 8, 8], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
func.func @set_encoding_dynamic() attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
} {
  %c0 = arith.constant 0 : index
  %d0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %d1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%d0, %d1}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding_lhs>>{%d0, %d1}
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%d0, %d1} -> tensor<?x?xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_lhs>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : tensor<?x?xf32, #encoding_lhs>
      -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding_lhs>>{%d0, %d1}
  return
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//       CHECK: func @set_encoding_dynamic()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 0.0
//   CHECK-DAG:   %[[D0:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[D1:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//       CHECK:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//   CHECK-DAG:   %[[TILED_D0:.+]] = affine.apply #[[MAP0]]()[%[[D0]]]
//   CHECK-DAG:   %[[TILED_D1:.+]] = affine.apply #[[MAP1]]()[%[[D1]]]
//   CHECK-DAG:   %[[OUTPUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//  CHECK-SAME:       !flow.dispatch.tensor<writeonly:tensor<?x?x8x4xf32>>{%[[TILED_D0]], %[[TILED_D1]]}
//       CHECK:   %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]]
//       CHECK:   %[[EMPTY:.+]] = tensor.empty
//       CHECK:   %[[PACK:.+]] = linalg.pack
//  CHECK-SAME:       %[[INPUT]] padding_value(%[[CST]] : f32)
//  CHECK-SAME:       inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[EMPTY]]
//       CHECK:   flow.dispatch.tensor.store %[[PACK]], %[[OUTPUT_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_D0]], %[[TILED_D1]], 8, 4], strides = [1, 1, 1, 1]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
func.func @unset_encoding_dynamic() attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %d0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %d1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%d0, %d1}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%d0, %d1}
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%d0, %d1}
      -> tensor<?x?xf32, #encoding_lhs>
  %3 = iree_encoding.unset_encoding %2
      : tensor<?x?xf32, #encoding_lhs> -> tensor<?x?xf32>{%d0, %d1}
  %4 = tensor.extract_slice %3[0, 0] [%d0, %d1] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [%d0, %d1], strides = [1, 1]
      : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%d0, %d1}
  return
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//       CHECK: func @unset_encoding_dynamic()
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[D0:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//   CHECK-DAG:   %[[D1:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//   CHECK-DAG:   %[[TILED_D0:.+]] = affine.apply #[[MAP0]]()[%[[D0]]]
//   CHECK-DAG:   %[[TILED_D1:.+]] = affine.apply #[[MAP1]]()[%[[D1]]]
//   CHECK-DAG:   %[[INPUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
//  CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x4xf32>>{%[[TILED_D0]], %[[TILED_D1]]}
//   CHECK-DAG:   %[[OUTPUT_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
//       CHECK:   %[[INPUT:.+]] = flow.dispatch.tensor.load %[[INPUT_BINDING]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_D0]], %[[TILED_D1]], 8, 4], strides = [1, 1, 1, 1]
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[D0]], %[[D1]])
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[INPUT]]
//  CHECK-SAME:       inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[EMPTY]]
//   CHECK-DAG:   flow.dispatch.tensor.store %[[UNPACK]], %[[OUTPUT_BINDING]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 3, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
func.func @matmul_lowering_f32f32f32_generic() attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
} {
  %c0 = arith.constant 0 : index
  %M = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %N = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %K = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_lhs>>{%M, %K}
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0)
      : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding_rhs>>{%K, %N}
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0)
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
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 8)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0] -> (s0 ceildiv 4)>
//      CHECK: func @matmul_lowering_f32f32f32_generic()
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[M:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(0)
//  CHECK-DAG:   %[[N:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(1)
//  CHECK-DAG:   %[[K:.+]] = hal.interface.constant.load layout({{.+}}) ordinal(2)
//  CHECK-DAG:   %[[TILED_M:.+]] = affine.apply #[[MAP0]]()[%[[M]]]
//  CHECK-DAG:   %[[TILED_K:.+]] = affine.apply #[[MAP1]]()[%[[K]]]
//      CHECK:   %[[LHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x4xf32>>{%[[TILED_M]], %[[TILED_K]]}
//      CHECK:   %[[TILED_N:.+]] = affine.apply #[[MAP0]]()[%[[N]]]
//      CHECK:   %[[RHS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
// CHECK-SAME:       !flow.dispatch.tensor<readonly:tensor<?x?x8x4xf32>>{%[[TILED_N]], %[[TILED_K]]}
//      CHECK:   %[[OUTS_BINDING:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(2)
// CHECK-SAME:       !flow.dispatch.tensor<readwrite:tensor<?x?x8x8xf32>>{%[[TILED_M]], %[[TILED_N]]}
//      CHECK:   %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_N]], %[[TILED_K]], 8, 4], strides = [1, 1, 1, 1]
//      CHECK:   %[[OUTS:.+]] = flow.dispatch.tensor.load %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   flow.dispatch.tensor.store %[[MMT4D]], %[[OUTS_BINDING]]
// CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [%[[TILED_M]], %[[TILED_N]], 8, 8], strides = [1, 1, 1, 1]
