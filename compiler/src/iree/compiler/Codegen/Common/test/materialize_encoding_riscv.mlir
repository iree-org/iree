// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding))" --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f32f32_riscv(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv32-xyz-xyz", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %acc, %c0 : tensor<?x?xf32>
  %N = tensor.dim %acc, %c1 : tensor<?x?xf32>
  %K = tensor.dim %lhs, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %lhs encoding_dims{%M, %N, %K} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_lhs>
  %1 = iree_encoding.set_encoding %rhs encoding_dims{%M, %N, %K} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_rhs>
  %2 = iree_encoding.set_encoding %acc encoding_dims{%M, %N, %K} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_result>
  %3 = linalg.matmul
      ins(%0, %1 : tensor<?x?xf32, #encoding_lhs>,
                   tensor<?x?xf32, #encoding_rhs>)
      outs(%2 : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  %4 = iree_encoding.unset_encoding %3 encoding_dims{%M, %N, %K} : tensor<?x?xf32, #encoding_result> -> tensor<?x?xf32>{%M, %N}
  return %4 : tensor<?x?xf32>
}
// RISC-V targets does not implement data-tiling yet.
// CHECK-LABEL: func @matmul_lowering_f32f32f32_riscv
//       CHECK:   %[[RES:.+]] = linalg.matmul
//       CHECK:   return %[[RES]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i8i32_riscv32_ukernel(
    %lhs: tensor<?x?xi8, #encoding_lhs>,
    %rhs: tensor<?x?xi8, #encoding_rhs>,
    %result: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple="riscv32-xyz-xyz", ukernels = "all", iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>
} {
  %out = linalg.matmul
      ins(%lhs, %rhs : tensor<?x?xi8, #encoding_lhs>,
                       tensor<?x?xi8, #encoding_rhs>)
      outs(%result : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %out : tensor<?x?xi32, #encoding_result>
}
// CHECK-LABEL: func @matmul_lowering_i8i8i32_riscv32_ukernel(
// CHECK-SAME:    %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x4xi8>
// CHECK-SAME:    %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x8x4xi8>
// CHECK-SAME:    %[[ACC:[a-zA-Z0-9]+]]: tensor<?x?x8x8xi32>
// CHECK:         %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:      ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:      outs(%[[ACC]]
// CHECK:         return %[[MMT4D]]
