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
