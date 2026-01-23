// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding{test-gpu-encoding-resolver=gpu_data_tiling}))" \
// RUN:   --iree-gpu-test-target=gfx1100 \
// RUN:   --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_WMMAR3_F32_16x16x16_F16(
  %lhs: tensor<?x?xf16, #encoding_lhs>,
  %rhs: tensor<?x?xf16, #encoding_rhs>,
  %acc: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result> {
  %result = linalg.matmul
    ins(%lhs, %rhs : tensor<?x?xf16, #encoding_lhs>, tensor<?x?xf16, #encoding_rhs>)
    outs(%acc : tensor<?x?xf32, #encoding_result>)
    -> tensor<?x?xf32, #encoding_result>
  return %result : tensor<?x?xf32, #encoding_result>
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK:     func.func @matmul_lowering_WMMAR3_F32_16x16x16_F16(
// CHECK-SAME:   %[[LHS:.+]]: tensor<?x?x2x2x1x16x16xf16>, %[[RHS:.+]]: tensor<?x?x2x2x1x16x16xf16>
// CHECK-SAME:   %[[ACC:.+]]: tensor<?x?x2x2x2x2x8x2x16xf32>
// CHECK-SAME: ) -> tensor<?x?x2x2x2x2x8x2x16xf32>
// CHECK:       %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = WMMAR3_F32_16x16x16_F16, intrinsics_m = 2, subgroups_m = 2, intrinsics_n = 2, subgroups_n = 2, operands_interleaving_intrinsics_k = [0, 1]>
// CHECK:       return %[[MMA]]

// -----

// Test that linalg.index ops are correctly remapped to account for packed dimensions.
// The output tensor is packed as tensor<8x29x8x2x16xf32> with swizzle permutation.
// Original dim 0 (M) maps to: d0 * 16 + (d3 + d2 * 2)
// Original dim 1 (N) maps to: d1 * 16 + d4

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [123, 456, 789]>
func.func @generic_with_linalg_index(
    %arg0: tensor<123x456xf32, #encoding>
) -> tensor<123x456xf32, #encoding> {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<123x456xf32, #encoding>
  %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<123x456xf32, #encoding>) outs(%0 : tensor<123x456xf32, #encoding>) {
  ^bb0(%in: f32, %out: f32):
    %2 = linalg.index 0 : index
    %3 = linalg.index 1 : index
    %4 = arith.cmpi sge, %2, %3 : index
    %5 = arith.select %4, %cst, %cst_0 : f32
    %6 = arith.addf %in, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<123x456xf32, #encoding>
  return %1 : tensor<123x456xf32, #encoding>
}
// CHECK-LABEL: func.func @generic_with_linalg_index(
// CHECK-SAME:    %[[ARG0:.+]]: tensor<8x29x8x2x16xf32>
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK:       linalg.generic
// CHECK-SAME:    ins(%[[ARG0]] : tensor<8x29x8x2x16xf32>)
// CHECK:       ^bb0(%[[IN:.+]]: f32, %[[OUT:.+]]: f32):
// CHECK-DAG:     %[[D0:.+]] = linalg.index 0 : index
// CHECK-DAG:     %[[D3:.+]] = linalg.index 3 : index
// CHECK-DAG:     %[[D2:.+]] = linalg.index 2 : index
// CHECK:         %[[MUL0:.+]] = arith.muli %[[D2]], %[[C2]] : index
// CHECK:         %[[INNER0:.+]] = arith.addi %[[D3]], %[[MUL0]] : index
// CHECK:         %[[OUTER0:.+]] = arith.muli %[[D0]], %[[C16]] : index
// CHECK:         %[[IDX0:.+]] = arith.addi %[[OUTER0]], %[[INNER0]] : index
// CHECK-DAG:     %[[D1:.+]] = linalg.index 1 : index
// CHECK-DAG:     %[[D4:.+]] = linalg.index 4 : index
// CHECK:         %[[OUTER1:.+]] = arith.muli %[[D1]], %[[C16]] : index
// CHECK:         %[[IDX1:.+]] = arith.addi %[[OUTER1]], %[[D4]] : index
// CHECK:         arith.cmpi sge, %[[IDX0]], %[[IDX1]] : index
