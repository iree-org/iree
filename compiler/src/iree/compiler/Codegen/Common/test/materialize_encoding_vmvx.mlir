// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-device-encoding),canonicalize,cse)" --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_i8i8i32_vmvx_ukernel(
  %lhs: tensor<?x?xi8, #encoding_lhs>,
  %rhs: tensor<?x?xi8, #encoding_rhs>,
  %result: tensor<?x?xi32, #encoding_result>
) -> tensor<?x?xi32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "all", iree.encoding.resolver = #iree_cpu.vmvx_encoding_resolver<>}>
} {
  %0 = linalg.matmul
      ins(%lhs, %rhs : tensor<?x?xi8, #encoding_lhs>,
                       tensor<?x?xi8, #encoding_rhs>)
      outs(%result : tensor<?x?xi32, #encoding_result>)
      -> tensor<?x?xi32, #encoding_result>
  return %0 : tensor<?x?xi32, #encoding_result>
}
//      CHECK: func @matmul_lowering_i8i8i32_vmvx_ukernel(
// CHECK-SAME:   %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x?x?xi8>
// CHECK-SAME:   %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x?x?xi8>
// CHECK-SAME:   %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x?x?xi32>
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   return %[[MMT4D]]

// -----

#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map2, #map3, #map4], iteration_sizes = [1, 3, 2]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map2, #map3, #map4], iteration_sizes = [1, 3, 2]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map2, #map3, #map4], iteration_sizes = [1, 3, 2]>
func.func @fill_matmul(
    %lhs: tensor<1x2xf32, #encoding_lhs>,
    %rhs: tensor<2x3xf32, #encoding_rhs>
) -> tensor<1x3xf32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_cpu.vmvx_encoding_resolver<>}>
} {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<1x3xf32, #encoding_result>
  %filled = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x3xf32, #encoding_result>) -> tensor<1x3xf32, #encoding_result>
  %result = linalg.matmul ins(%lhs, %rhs : tensor<1x2xf32, #encoding_lhs>, tensor<2x3xf32, #encoding_rhs>) outs(%filled : tensor<1x3xf32, #encoding_result>) -> tensor<1x3xf32, #encoding_result>
  return %result : tensor<1x3xf32, #encoding_result>
}
//      CHECK: func.func @fill_matmul
// CHECK-SAME:   %[[LHS:.+]]: tensor<1x1x1x4xf32>
// CHECK-SAME:   %[[RHS:.+]]: tensor<1x1x8x4xf32>
//  CHECK-DAG:   %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
//      CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<1x1x1x8xf32>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:     ins(%[[ZERO]]
// CHECK-SAME:     outs(%[[EMPTY]]
//      CHECK:   %[[MATMUL:.+]] = linalg.mmt4d
// CHECK-SAME:     ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:     outs(%[[FILL]]
//      CHECK:   return %[[MATMUL]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @set_encoding_dynamic(%input: tensor<?x?xf32>, %m: index, %n: index, %k: index) -> tensor<?x?xf32, #encoding_lhs> attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_cpu.vmvx_encoding_resolver<>}>
} {
  %0 = iree_encoding.set_encoding %input encoding_dims{%m, %n, %k} : tensor<?x?xf32> -> tensor<?x?xf32, #encoding_lhs>
  return %0 : tensor<?x?xf32, #encoding_lhs>
}
//       CHECK: func @set_encoding_dynamic(
//  CHECK-SAME:   %[[INPUT:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:   %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[EMPTY:.+]] = tensor.empty
//       CHECK:   %[[PACK:.+]] = linalg.pack
//  CHECK-SAME:       %[[INPUT]] padding_value(%[[CST]] : f32)
//  CHECK-SAME:       inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[EMPTY]]
//       CHECK:   return %[[PACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @unset_encoding_dynamic(%input: tensor<?x?xf32, #encoding_lhs>, %d0: index, %d1: index, %m: index, %n: index, %k: index) -> tensor<?x?xf32> attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_cpu.vmvx_encoding_resolver<>}>
} {
  %0 = iree_encoding.unset_encoding %input encoding_dims{%m, %n, %k} : tensor<?x?xf32, #encoding_lhs> -> tensor<?x?xf32>{%d0, %d1}
  return %0 : tensor<?x?xf32>
}
//       CHECK: func @unset_encoding_dynamic(
//  CHECK-SAME:   %[[INPUT:[a-zA-Z0-9]+]]: tensor<?x?x8x4xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[INPUT]]
//  CHECK-SAME:       inner_dims_pos = [0, 1] inner_tiles = [8, 4] into %[[EMPTY]]
//       CHECK:   return %[[UNPACK]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding_lhs = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_rhs = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
#encoding_result = #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, ?, ?]>
func.func @matmul_lowering_f32f32f32_generic(
    %lhs: tensor<?x?xf32, #encoding_lhs>,
    %rhs: tensor<?x?xf32, #encoding_rhs>,
    %result: tensor<?x?xf32, #encoding_result>
) -> tensor<?x?xf32, #encoding_result> attributes {
  hal.executable.target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_cpu.vmvx_encoding_resolver<>}>
} {
  %0 = linalg.matmul
      ins(%lhs, %rhs : tensor<?x?xf32, #encoding_lhs>, tensor<?x?xf32, #encoding_rhs>)
      outs(%result : tensor<?x?xf32, #encoding_result>)
      -> tensor<?x?xf32, #encoding_result>
  return %0 : tensor<?x?xf32, #encoding_result>
}
//      CHECK: func @matmul_lowering_f32f32f32_generic(
// CHECK-SAME:     %[[LHS:[a-zA-Z0-9]+]]: tensor<?x?x8x4xf32>
// CHECK-SAME:     %[[RHS:[a-zA-Z0-9]+]]: tensor<?x?x8x4xf32>
// CHECK-SAME:     %[[OUTS:[a-zA-Z0-9]+]]: tensor<?x?x8x8xf32>
//      CHECK:   %[[MMT4D:.+]] = linalg.mmt4d
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   return %[[MMT4D]]
