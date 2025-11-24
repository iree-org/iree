// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding{test-gpu-encoding-resolver=gpu_data_tiling}))" \
// RUN:   --iree-gpu-test-target=gfx908 \
// RUN:   --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_MFMA_i32_16x16x16_i8(
  %lhs: tensor<?x?xi8, #encoding_lhs>,
  %rhs: tensor<?x?xi8, #encoding_rhs>,
  %acc: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> {
  %result = linalg.matmul
    ins(%lhs, %rhs : tensor<?x?xi8, #encoding_lhs>,
                     tensor<?x?xi8, #encoding_rhs>)
    outs(%acc : tensor<?x?xi32, #encoding_result>)
    -> tensor<?x?xi32, #encoding_result>
  return %result : tensor<?x?xi32, #encoding_result>
}
// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @matmul_lowering_MFMA_i32_16x16x16_i8(
// CHECK-SAME:   %[[LHS:.+]]: tensor<?x?x4x4x16x4x4xi8>, %[[RHS:.+]]: tensor<?x?x4x2x4x16x4x4xi8>
// CHECK-SAME:   %[[ACC:.+]]: tensor<?x?x4x4x2x4x16x4xi32>
// CHECK-SAME: ) -> tensor<?x?x4x4x2x4x16x4xi32>
// CHECK:       %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:    indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]],
// CHECK-SAME:    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_I32_16x16x16_I8, intrinsics_m = 4, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 4, operands_interleaving_intrinsics_k = [0, 1]>
// CHECK:       return %[[MMA]]
