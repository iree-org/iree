// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding{test-gpu-encoding-resolver=gpu_data_tiling}))" \
// RUN:   --iree-gpu-test-target=gfx90a \
// RUN:   --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_MFMA_f32_16x16x8_bf16(
  %lhs: tensor<?x?xbf16, #encoding_lhs>,
  %rhs: tensor<?x?xbf16, #encoding_rhs>,
  %acc: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result> {
  %result = linalg.matmul
    ins(%lhs, %rhs: tensor<?x?xbf16, #encoding_lhs>,
                    tensor<?x?xbf16, #encoding_rhs>)
    outs(%acc: tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  return %result : tensor<?x?xf32, #encoding_result>
}
// CHECK-LABEL: func.func @matmul_lowering_MFMA_f32_16x16x8_bf16(
// CHECK-SAME:   %[[LHS:.+]]: tensor<?x?x4x4x16x4x2xbf16>, %[[RHS:.+]]: tensor<?x?x4x2x4x16x4x2xbf16>
// CHECK-SAME:   %[[ACC:.+]]: tensor<?x?x4x4x2x4x16x4xf32>
// CHECK-SAME: ) -> tensor<?x?x4x4x2x4x16x4xf32>
// CHECK:       %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:    indexing_maps = [#{{[^,]+}}, #{{[^,]+}}, #{{[^]]+}}],
// CHECK-SAME:    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F32_16x16x8_BF16, intrinsics_m = 4, intrinsics_n = 2, subgroups_n = 4, intrinsics_k = 4, operands_interleaving_intrinsics_k = [0, 1]>
// CHECK:       return %[[MMA]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f64, f64, f64], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f64, f64, f64], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f64, f64, f64], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_MFMA_f64_16x16x4_f64(
    %lhs: tensor<?x?xf64, #encoding_lhs>,
    %rhs: tensor<?x?xf64, #encoding_rhs>,
    %acc: tensor<?x?xf64, #encoding_result>
) -> tensor<?x?xf64, #encoding_result> {
  %result = linalg.matmul
      ins(%lhs, %rhs : tensor<?x?xf64, #encoding_lhs>,
                       tensor<?x?xf64, #encoding_rhs>)
      outs(%acc : tensor<?x?xf64, #encoding_result>)
      -> tensor<?x?xf64, #encoding_result>
  return %result : tensor<?x?xf64, #encoding_result>
}
// CHECK-LABEL: func.func @matmul_lowering_MFMA_f64_16x16x4_f64(
// CHECK-SAME:     %[[LHS:.+]]: tensor<?x?x2x2x4x16x2xf64>, %[[RHS:.+]]: tensor<?x?x2x2x4x16x2xf64>
// CHECK-SAME:     %[[ACC:.+]]: tensor<?x?x2x2x2x2x4x4x16xf64>
// CHECK-SAME:   ) -> tensor<?x?x2x2x2x2x4x4x16xf64>
// CHECK:       %[[MMA:.+]] = iree_codegen.inner_tiled ins(%[[LHS]], %[[RHS]]) outs(%[[ACC]])
// CHECK-SAME:    indexing_maps = [#{{[^,]+}}, #{{[^,]+}}, #{{[^]]+}}],
// CHECK-SAME:    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]
// CHECK-SAME:    kind = #iree_gpu.data_tiled_mma_layout<intrinsic = MFMA_F64_16x16x4_F64, intrinsics_m = 2, subgroups_m = 2, intrinsics_n = 2, subgroups_n = 2, intrinsics_k = 2, operands_interleaving_intrinsics_k = [0, 1]>
// CHECK:       return %[[MMA]]
