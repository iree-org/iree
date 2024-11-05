// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-encoding-into-nop))" -split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
func.func @pack_unpack_gemm_lhs(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_lhs>
  %1 = iree_encoding.unset_encoding %0 : tensor<?x?xf32, #encoding_lhs> -> tensor<?x?xf32>{%d0, %d1}
  return %1 : tensor<?x?xf32>
}
//      CHECK: func @pack_unpack_gemm_lhs(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//      CHECK:   return %[[ARG0]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
func.func @gemm_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %arg2, %c0 : tensor<?x?xf32>
  %N = tensor.dim %arg2, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_lhs>
  %1 = iree_encoding.set_encoding %arg1 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_rhs>
  %2 = iree_encoding.set_encoding %arg2 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_result>
  %3 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #encoding_lhs>, tensor<?x?xf32, #encoding_rhs>)
      outs(%2 : tensor<?x?xf32, #encoding_result>) -> tensor<?x?xf32, #encoding_result>
  %4 = iree_encoding.unset_encoding %3 : tensor<?x?xf32, #encoding_result> -> tensor<?x?xf32>{%M, %N}
  return %4 : tensor<?x?xf32>
}
//      CHECK: func @gemm_dynamic(
// CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[DEST:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//      CHECK:   %[[RES:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[DEST]] :
//      CHECK:   return %[[RES]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
func.func @gemm_fill_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_lhs>
  %1 = iree_encoding.set_encoding %arg1 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_rhs>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #encoding_result>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #encoding_lhs>, tensor<?x?xf32, #encoding_rhs>)
      outs(%3 : tensor<?x?xf32, #encoding_result>) -> tensor<?x?xf32, #encoding_result>
  %5 = iree_encoding.unset_encoding %4 : tensor<?x?xf32, #encoding_result> -> tensor<?x?xf32>{%d0, %d1}
  return %5 : tensor<?x?xf32>
}
//      CHECK: func @gemm_fill_dynamic(
// CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[LHS]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[RHS]], %[[C1]]
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[D0]], %[[D1]]) : tensor<?x?xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   %[[RES:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[FILL]] :
//      CHECK:   return %[[RES]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
func.func @batch_matmul(%arg0 : tensor<128x80x32xf32>, %arg1 : tensor<128x32x320xf32>, %arg2 : tensor<128x80x320xf32>) -> tensor<128x80x320xf32> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<128x80x32xf32> -> tensor<128x80x32xf32, #encoding_lhs>
  %1 = iree_encoding.set_encoding %arg1 : tensor<128x32x320xf32> -> tensor<128x32x320xf32, #encoding_rhs>
  %2 = iree_encoding.set_encoding %arg2 : tensor<128x80x320xf32> -> tensor<128x80x320xf32, #encoding_result>
  %3 = linalg.batch_matmul ins(%0, %1 : tensor<128x80x32xf32, #encoding_lhs>, tensor<128x32x320xf32, #encoding_rhs>)
      outs(%2 : tensor<128x80x320xf32, #encoding_result>) -> tensor<128x80x320xf32, #encoding_result>
  %4 = iree_encoding.unset_encoding %3 : tensor<128x80x320xf32, #encoding_result> -> tensor<128x80x320xf32>
  return %4 : tensor<128x80x320xf32>
}
//      CHECK: func @batch_matmul(
// CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<128x80x32xf32>
// CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: tensor<128x32x320xf32>
// CHECK-SAME:     %[[DEST:[a-zA-Z0-9]+]]: tensor<128x80x320xf32>
//      CHECK:   %[[RES:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[DEST]] :
//      CHECK:   return %[[RES]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
func.func @batch_matmul_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>, %arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %B = tensor.dim %arg2, %c0 : tensor<?x?x?xf32>
  %M = tensor.dim %arg2, %c1 : tensor<?x?x?xf32>
  %N = tensor.dim %arg2, %c2 : tensor<?x?x?xf32>
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding_lhs>
  %1 = iree_encoding.set_encoding %arg1 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding_rhs>
  %2 = iree_encoding.set_encoding %arg2 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding_result>
  %3 = linalg.batch_matmul ins(%0, %1 : tensor<?x?x?xf32, #encoding_lhs>, tensor<?x?x?xf32, #encoding_rhs>)
      outs(%2 : tensor<?x?x?xf32, #encoding_result>) -> tensor<?x?x?xf32, #encoding_result>
  %4 = iree_encoding.unset_encoding %3 : tensor<?x?x?xf32, #encoding_result> -> tensor<?x?x?xf32>{%B, %M, %N}
  return %4 : tensor<?x?x?xf32>
}
//      CHECK: func @batch_matmul_dynamic(
// CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x?xf32>
// CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x?xf32>
// CHECK-SAME:     %[[DEST:[a-zA-Z0-9]+]]: tensor<?x?x?xf32>
//      CHECK:   %[[RES:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[DEST]] :
//      CHECK:   return %[[RES]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
func.func @batch_matmul_fill_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %d2 = tensor.dim %arg1, %c2 : tensor<?x?x?xf32>
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding_lhs>
  %1 = iree_encoding.set_encoding %arg1 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding_rhs>
  %2 = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32, #encoding_result>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?x?xf32, #encoding_result>)
      -> tensor<?x?x?xf32, #encoding_result>
  %4 = linalg.batch_matmul ins(%0, %1 : tensor<?x?x?xf32, #encoding_lhs>, tensor<?x?x?xf32, #encoding_rhs>)
      outs(%3 : tensor<?x?x?xf32, #encoding_result>) -> tensor<?x?x?xf32, #encoding_result>
  %5 = iree_encoding.unset_encoding %4 : tensor<?x?x?xf32, #encoding_result> -> tensor<?x?x?xf32>{%d0, %d1, %d2}
  return %5 : tensor<?x?x?xf32>
}
//      CHECK: func @batch_matmul_fill_dynamic(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG1]], %[[C2]]
//  CHECK-DAG:   %[[EMPTY:.+]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]]) : tensor<?x?x?xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:     outs(%[[EMPTY]] : tensor<?x?x?xf32>)
//      CHECK:   %[[RES:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[FILL]] :
//      CHECK:   return %[[RES]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]>
func.func @drop_encoding_for_hal_flow_ops_static() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x1xf32, #encoding_lhs>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1xf32>> -> tensor<1x1xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<1x1xf32> -> tensor<1x1xf32, #encoding_lhs>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [1, 1], strides = [1, 1] : tensor<1x1xf32, #encoding_lhs> -> !flow.dispatch.tensor<writeonly:tensor<1x1xf32, #encoding_lhs>>
  return
}
// CHECK-LABEL: func.func @drop_encoding_for_hal_flow_ops_static
// CHECK-DAG:     %[[IN:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0) {{.+}} : !flow.dispatch.tensor<readonly:tensor<1x1xf32>>
// CHECK-DAG:     %[[OUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1) {{.+}} : !flow.dispatch.tensor<writeonly:tensor<1x1xf32>>
// CHECK:         %[[LOAD:.+]] = flow.dispatch.tensor.load %[[IN]]
// CHECK:         flow.dispatch.tensor.store %[[LOAD]], %[[OUT]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 4, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]>
func.func @drop_encoding_for_hal_flow_ops_dynamic() {
  %c0 = arith.constant 0 : index
  %c32_i64 = arith.constant 32 : i64
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = hal.interface.constant.load layout(#pipeline_layout) ordinal(2) : i32
  %3 = hal.interface.constant.load layout(#pipeline_layout) ordinal(3) : i32
  %4 = arith.extui %0 : i32 to i64
  %5 = arith.extui %1 : i32 to i64
  %6 = arith.shli %5, %c32_i64 : i64
  %7 = arith.ori %4, %6 : i64
  %8 = arith.index_castui %7 : i64 to index
  %9 = arith.extui %2 : i32 to i64
  %10 = arith.extui %3 : i32 to i64
  %11 = arith.shli %10, %c32_i64 : i64
  %12 = arith.ori %9, %11 : i64
  %13 = arith.index_castui %12 : i64 to index
  %14 = flow.dispatch.workload.ordinal %8, 0 : index
  %15 = flow.dispatch.workload.ordinal %13, 1 : index
  %16 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<?x?xbf16>>{%14, %15}
  %17 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<?x?xbf16, #encoding_lhs>>{%14, %15}
  %18 = flow.dispatch.tensor.load %16, offsets = [0, 0], sizes = [%14, %15], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xbf16>>{%14, %15} -> tensor<?x?xbf16>
  %19 = iree_encoding.set_encoding %18 : tensor<?x?xbf16> -> tensor<?x?xbf16, #encoding_lhs>
  flow.dispatch.tensor.store %19, %17, offsets = [0, 0], sizes = [%14, %15], strides = [1, 1] : tensor<?x?xbf16, #encoding_lhs> -> !flow.dispatch.tensor<writeonly:tensor<?x?xbf16, #encoding_lhs>>{%14, %15}
  return
}
// CHECK-LABEL: func.func @drop_encoding_for_hal_flow_ops_dynamic
// CHECK-DAG:     %[[IN:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0) {{.+}} : !flow.dispatch.tensor<readonly:tensor<?x?xbf16>>
// CHECK-DAG:     %[[OUT:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1) {{.+}} : !flow.dispatch.tensor<writeonly:tensor<?x?xbf16>>
// CHECK:         %[[LOAD:.+]] = flow.dispatch.tensor.load %[[IN]]
// CHECK:         flow.dispatch.tensor.store %[[LOAD]], %[[OUT]]
