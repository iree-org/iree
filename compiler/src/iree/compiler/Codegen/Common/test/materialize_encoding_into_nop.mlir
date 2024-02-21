// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-encoding-into-nop))" -split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @pack_unpack_gemm_lhs(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %1 = iree_linalg_ext.unset_encoding %0 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//      CHECK: func @pack_unpack_gemm_lhs(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>
//      CHECK:   return %[[ARG0]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @pad_gemm(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250x500xf32>, %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %pad_value = arith.constant 0.0 : f32
  %pad_lhs = tensor.pad %arg0 low[0, 0] high[4, 2] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value : f32
    } : tensor<100x250xf32> to tensor<104x252xf32>
  %lhs = iree_linalg_ext.set_encoding %pad_lhs : tensor<104x252xf32> -> tensor<104x252xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %pad_rhs = tensor.pad %arg1 low[0, 0] high[2, 4] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value : f32
    } : tensor<250x500xf32> to tensor<252x504xf32>
  %rhs = iree_linalg_ext.set_encoding %pad_rhs : tensor<252x504xf32> -> tensor<252x504xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %pad_output = tensor.pad %arg2 low[0, 0] high[4, 4] {
    ^bb0(%b0: index, %b1 : index):
      tensor.yield %pad_value : f32
    } : tensor<100x500xf32> to tensor<104x504xf32>
  %output = iree_linalg_ext.set_encoding %pad_output : tensor<104x504xf32> -> tensor<104x504xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %gemm_packed = linalg.matmul ins(%lhs, %rhs : tensor<104x252xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>, tensor<252x504xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%output : tensor<104x504xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<104x504xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %gemm = iree_linalg_ext.unset_encoding %gemm_packed : tensor<104x504xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> tensor<104x504xf32>
  %result = tensor.extract_slice %gemm[0, 0] [100, 500] [1, 1] : tensor<104x504xf32> to tensor<100x500xf32>
  return %result : tensor<100x500xf32>
}
//      CHECK: func @pad_gemm(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[CST:.+]] = arith.constant 0.0
//  CHECK-DAG:   %[[LHS:.+]] = tensor.pad %[[ARG0]]
//  CHECK-DAG:   %[[RHS:.+]] = tensor.pad %[[ARG1]]
//  CHECK-DAG:   %[[DEST:.+]] = tensor.pad %[[ARG2]]
//      CHECK:   %[[GEMM:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[DEST]] :
//      CHECK:   %[[RES:.+]] = tensor.extract_slice %[[GEMM]]
//      CHECK:   return %[[RES]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @gemm_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %1 = iree_linalg_ext.set_encoding %arg1 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %2 = iree_linalg_ext.set_encoding %arg2 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %3 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>, tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%2 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = iree_linalg_ext.unset_encoding %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?xf32>
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
func.func @gemm_fill_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %1 = iree_linalg_ext.set_encoding %arg1 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %2 = tensor.empty(%d0, %d1) : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = linalg.matmul ins(%0, %1 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>, tensor<?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%3 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = iree_linalg_ext.unset_encoding %4 : tensor<?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?xf32>
  return %5 : tensor<?x?xf32>
}
//      CHECK: func @gemm_fill_dynamic(
// CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
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
func.func @batch_matmul(%arg0 : tensor<128x80x32xf32>, %arg1 : tensor<128x32x320xf32>, %arg2 : tensor<128x80x320xf32>) -> tensor<128x80x320xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<128x80x32xf32> -> tensor<128x80x32xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %1 = iree_linalg_ext.set_encoding %arg1 : tensor<128x32x320xf32> -> tensor<128x32x320xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %2 = iree_linalg_ext.set_encoding %arg2 : tensor<128x80x320xf32> -> tensor<128x80x320xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %3 = linalg.batch_matmul ins(%0, %1 : tensor<128x80x32xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>, tensor<128x32x320xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%2 : tensor<128x80x320xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<128x80x320xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = iree_linalg_ext.unset_encoding %3 : tensor<128x80x320xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> tensor<128x80x320xf32>
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
func.func @batch_matmul_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>, %arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %1 = iree_linalg_ext.set_encoding %arg1 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %2 = iree_linalg_ext.set_encoding %arg2 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %3 = linalg.batch_matmul ins(%0, %1 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>, tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%2 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = iree_linalg_ext.unset_encoding %3 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?x?xf32>
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
func.func @batch_matmul_fill_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 0.0 : f32
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %d2 = tensor.dim %arg1, %c2 : tensor<?x?x?xf32>
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %1 = iree_linalg_ext.set_encoding %arg1 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %2 = tensor.empty(%d0, %d1, %d2) : tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %4 = linalg.batch_matmul ins(%0, %1 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = LHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>, tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RHS, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>)
      outs(%3 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>) -> tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>>
  %5 = iree_linalg_ext.unset_encoding %4 : tensor<?x?x?xf32, #iree_linalg_ext.encoding<role = RESULT, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>> -> tensor<?x?x?xf32>
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
