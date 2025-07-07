// RUN: iree-opt --split-input-file %s | FileCheck %s

#encoding = #iree_encoding.testing<>
func.func @set_encoding_ops(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  return %0 : tensor<?x?xf32, #encoding>
}
//      CHECK: #[[ENCODING:.+]] = #iree_encoding.testing<>
//      CHECK: func.func @set_encoding_ops
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//      CHECK:   iree_encoding.set_encoding %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING]]>

// -----

#encoding = #iree_encoding.testing<>
func.func @set_encoding_ops_mixed_dynamic_static(%arg0: tensor<?x10xf32>) -> tensor<20x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x10xf32> -> tensor<20x?xf32, #encoding>
  return %0 : tensor<20x?xf32, #encoding>
}
//      CHECK: #[[ENCODING:.+]] = #iree_encoding.testing<>
//      CHECK: func.func @set_encoding_ops_mixed_dynamic_static
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//      CHECK:   iree_encoding.set_encoding %[[ARG0]] : tensor<?x10xf32> -> tensor<20x?xf32, #[[ENCODING]]>

// -----

#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
#encoding1 = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [i8, i8, i32]>
#encoding2 = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f16, f16, f32]>
#encoding3 = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f16, f16, f16]>
#encoding4 = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [bf16, bf16, f32]>
#encoding5 = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [bf16, bf16, bf16]>
func.func @set_encoding_with_batch_matmul_user(%arg0: tensor<?x?x?xf32>) {
  iree_encoding.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding>
  iree_encoding.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding1>
  iree_encoding.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding2>
  iree_encoding.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding3>
  iree_encoding.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding4>
  iree_encoding.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #encoding5>
  return
}
//  CHECK-DAG: #[[ENCODING0:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
//  CHECK-DAG: #[[ENCODING1:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [i8, i8, i32]>
//  CHECK-DAG: #[[ENCODING2:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f16, f16, f32]>
//  CHECK-DAG: #[[ENCODING3:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f16, f16, f16]>
//  CHECK-DAG: #[[ENCODING4:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [bf16, bf16, f32]>
//  CHECK-DAG: #[[ENCODING5:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [bf16, bf16, bf16]>
//      CHECK: func.func @set_encoding_with_batch_matmul_user
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
//      CHECK:   iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #[[ENCODING0]]>
//      CHECK:   iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #[[ENCODING1]]>
//      CHECK:   iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #[[ENCODING2]]>
//      CHECK:   iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #[[ENCODING3]]>
//      CHECK:   iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #[[ENCODING4]]>
//      CHECK:   iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #[[ENCODING5]]>

// -----

#encoding = #iree_encoding.testing<>
func.func @unset_encoding_fully_static(%arg0: tensor<3x5xf32, #encoding>) -> tensor<3x5xf32> {
  %0 = iree_encoding.unset_encoding %arg0 : tensor<3x5xf32, #encoding> -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}
//      CHECK: #[[ENCODING:.+]] = #iree_encoding.testing<>
//      CHECK: func.func @unset_encoding_fully_static
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<3x5xf32, #[[ENCODING]]
//      CHECK:   iree_encoding.unset_encoding %[[ARG0]] : tensor<3x5xf32, #[[ENCODING]]> -> tensor<3x5xf32>

// -----

#encoding = #iree_encoding.testing<>
func.func @unset_encoding_fully_dynamic(%arg0: tensor<?x?xf32, #encoding>, %d0 : index, %d1 : index) -> tensor<?x?xf32> {
  %0 = iree_encoding.unset_encoding %arg0 : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>{%d0, %d1}
  return %0 : tensor<?x?xf32>
}
//      CHECK: #[[ENCODING:.+]] = #iree_encoding.testing<>
//      CHECK: func.func @unset_encoding_fully_dynamic
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32, #[[ENCODING]]
// CHECK-SAME:      %[[D0:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[D1:[a-zA-Z0-9]+]]
//      CHECK:   iree_encoding.unset_encoding %[[ARG0]] : tensor<?x?xf32, #[[ENCODING]]> -> tensor<?x?xf32>{%[[D0]], %[[D1]]}

// -----

#encoding = #iree_encoding.testing<>
func.func @unset_encoding_ops_mixed_dynamic_static(%arg0: tensor<10x?xf32, #encoding>, %d0 : index) -> tensor<?x20xf32> {
  %0 = iree_encoding.unset_encoding %arg0 : tensor<10x?xf32, #encoding> -> tensor<?x20xf32>{%d0}
  return %0 : tensor<?x20xf32>
}
//      CHECK: #[[ENCODING:.+]] = #iree_encoding.testing<>
//      CHECK: func.func @unset_encoding_ops_mixed_dynamic_static
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<10x?xf32, #[[ENCODING]]>
// CHECK-SAME:     %[[D0:[a-zA-Z0-9]+]]
//      CHECK:   iree_encoding.unset_encoding %[[ARG0]] : tensor<10x?xf32, #[[ENCODING]]> -> tensor<?x20xf32>{%[[D0]]}

// -----

#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
#encoding1 = #iree_encoding.encoding<operand_index = 1 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
#encoding2 = #iree_encoding.encoding<operand_index = 2 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
func.func @encoding_tensors_with_ops(%arg0 : tensor<?x?xf32>,
    %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %N = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  %1 = iree_encoding.set_encoding %arg1 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding1>
  %2 = iree_encoding.set_encoding %arg2 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding2>
  %3 = linalg.matmul
      ins(%0, %1 : tensor<?x?xf32, #encoding>, tensor<?x?xf32, #encoding1>)
      outs(%2 : tensor<?x?xf32, #encoding2>)
      -> tensor<?x?xf32, #encoding2>
  %4 = iree_encoding.unset_encoding %3 : tensor<?x?xf32, #encoding2> -> tensor<?x?xf32>{%M, %N}
  return %4 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[ENCODING0:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
//  CHECK-DAG: #[[ENCODING1:.+]] = #iree_encoding.encoding<operand_index = 1 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
//  CHECK-DAG: #[[ENCODING2:.+]] = #iree_encoding.encoding<operand_index = 2 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
//      CHECK:  func.func @encoding_tensors_with_ops
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:    %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:    %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:    %[[N:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:    %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:        tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING0]]>
//      CHECK:    %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:        tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING1]]>
//      CHECK:    %[[OUT:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:        tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING2]]>
//      CHECK:    %[[GEMM:.+]] = linalg.matmul
// CHECK-SAME:        ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:        outs(%[[OUT]]
//      CHECK:    %[[RESULT:.+]] = iree_encoding.unset_encoding %[[GEMM]] : tensor<?x?xf32, #[[ENCODING2]]> -> tensor<?x?xf32>{%[[M]], %[[N]]}
//      CHECK:    return %[[RESULT]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [[#map, #map3], #map1, #map2]>
func.func @set_encoding_ops_with_indexing_maps(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  return %0 : tensor<?x?xf32, #encoding>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG: #[[ENCODING:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32],
// CHECK-SAME:      user_indexing_maps = {{\[}}[#[[MAP0]], #[[MAP3]]], #[[MAP1]], #[[MAP2]]{{\]}}
//      CHECK: func.func @set_encoding_ops_with_indexing_maps(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]:
//      CHECK:   iree_encoding.set_encoding %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING]]>

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map, #map], iteration_sizes = [1, 1, 1]>
#encoding1 = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map, #map], iteration_sizes = [?, ?, ?]>
#encoding2 = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map, #map], iteration_sizes = [?, 1, ?]>
func.func @set_encoding_ops_with_iteration_sizes(%arg0: tensor<?x?xf32>) {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  %1 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding1>
  %2 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding2>
  return
}
//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//  CHECK-DAG: #[[ENCODING:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]], iteration_sizes = [1, 1, 1]>
//  CHECK-DAG: #[[ENCODING1:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]], iteration_sizes = [?, ?, ?]>
//  CHECK-DAG: #[[ENCODING2:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]], iteration_sizes = [?, 1, ?]>
//      CHECK:  func.func @set_encoding_ops_with_iteration_sizes(
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]:
//      CHECK:    iree_encoding.set_encoding %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING]]>
//      CHECK:    iree_encoding.set_encoding %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING1]]>
//      CHECK:    iree_encoding.set_encoding %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING2]]>

// -----

#encoding = #iree_encoding.specialization_resolver<123>
func.func @specialization_resolver(%arg0: tensor<?x?xf32, #encoding>) -> tensor<?x?xf32, #encoding> {
  return %arg0 : tensor<?x?xf32, #encoding>
}
//      CHECK: func.func @specialization_resolver(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf32, #iree_encoding.specialization_resolver<123>>
// CHECK         return %[[ARG0]]

// -----

#encoding = #iree_encoding.specialized<123>
func.func @specialized_without_type(%arg0: tensor<?x?xf32, #encoding>) -> tensor<?x?xf32, #encoding> {
  return %arg0 : tensor<?x?xf32, #encoding>
}
//      CHECK: func.func @specialized_without_type(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf32, #iree_encoding.specialized<123>>
// CHECK         return %[[ARG0]]

// -----

#encoding = #iree_encoding.specialized<123, tensor<?x?xf32>>
func.func @specialized_with_type(%arg0: tensor<?x?xf32, #encoding>) -> tensor<?x?xf32, #encoding> {
  return %arg0 : tensor<?x?xf32, #encoding>
}
//      CHECK: func.func @specialized_with_type(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf32, #iree_encoding.specialized<123, tensor<?x?xf32>>>
// CHECK         return %[[ARG0]]

// -----

#encoding = #iree_encoding.testing<>
func.func @testing_without_layouts(%arg0: tensor<?x?xf32, #encoding>) -> tensor<?x?xf32, #encoding> {
  return %arg0 : tensor<?x?xf32, #encoding>
}
//      CHECK: #[[ENCODING:.+]] = #iree_encoding.testing<>
//      CHECK: func.func @testing_without_layouts(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf32, #[[ENCODING]]>
// CHECK         return %[[ARG0]]

// -----

#encoding = #iree_encoding.testing<[#iree_encoding.specialization_resolver<123>]>
func.func @testing_with_layouts(%arg0: tensor<?x?xf32, #encoding>) -> tensor<?x?xf32, #encoding> {
  return %arg0 : tensor<?x?xf32, #encoding>
}
//      CHECK: #[[ENCODING:.+]] = #iree_encoding.testing<[#iree_encoding.specialization_resolver<123>]>
//      CHECK: func.func @testing_with_layouts(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf32, #[[ENCODING]]>
// CHECK         return %[[ARG0]]

// -----

#encoding = #iree_encoding.matmul_k<k_dims = [1]>
func.func @matmul_k_encoding(%arg0: tensor<?x?xf32, #encoding>) -> tensor<?x?xf32, #encoding> {
  return %arg0 : tensor<?x?xf32, #encoding>
}
//      CHECK: #[[ENCODING:.+]] = #iree_encoding.matmul_k<k_dims = [1]>
//      CHECK: func.func @matmul_k_encoding(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf32, #[[ENCODING]]>
// CHECK         return %[[ARG0]]

// -----

#encoding = #iree_encoding.layout<[#iree_encoding.padding<[0, 64]>]>
func.func @layout_encoding(%arg0: tensor<?x?xf32, #encoding>) -> tensor<?x?xf32, #encoding> {
  return %arg0 : tensor<?x?xf32, #encoding>
}
//      CHECK: #[[ENCODING:.+]] = #iree_encoding.layout<[#iree_encoding.padding<[0, 64]>]>
//      CHECK: func.func @layout_encoding(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf32, #[[ENCODING]]>
// CHECK         return %[[ARG0]]

// -----

#encoding = #iree_encoding.layout<[#iree_encoding.padding<[0, ?, 64]>]>
func.func @dynamic_layout_encoding(%arg0: tensor<?x?x?xf32, #encoding>) -> tensor<?x?x?xf32, #encoding> {
  return %arg0 : tensor<?x?x?xf32, #encoding>
}
//      CHECK: #[[ENCODING:.+]] = #iree_encoding.layout<[#iree_encoding.padding<[0, ?, 64]>]>
//      CHECK: func.func @dynamic_layout_encoding(%[[ARG0:.+]]: tensor<?x?x?xf32, #[[ENCODING]]>)

// -----

#encoding = #iree_encoding.identity
func.func @identity_encoding(%arg0: tensor<?x?xf32, #encoding>) -> tensor<?x?xf32, #encoding> {
  return %arg0 : tensor<?x?xf32, #encoding>
}
//      CHECK: func.func @identity_encoding(%[[ARG0:.+]]: tensor<?x?xf32, #iree_encoding.identity>
