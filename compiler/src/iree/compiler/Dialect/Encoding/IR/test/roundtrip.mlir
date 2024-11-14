// RUN: iree-opt --split-input-file %s | FileCheck %s

#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
func.func @set_encoding_ops(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  return %0 : tensor<?x?xf32, #encoding>
}
// CHECK:      #[[ENCODING:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
// CHECK:      func.func @set_encoding_ops
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:        iree_encoding.set_encoding %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING]]>

// -----

#encoding = #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>
func.func @set_encoding_ops_mixed_dynamic_static(%arg0: tensor<?x10xf32>) -> tensor<20x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x10xf32> -> tensor<20x?xf32, #encoding>
  return %0 : tensor<20x?xf32, #encoding>
}
// CHECK:       #[[ENCODING:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32]>
// CHECK:       func.func @set_encoding_ops_mixed_dynamic_static
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         iree_encoding.set_encoding %[[ARG0]] : tensor<?x10xf32> -> tensor<20x?xf32, #[[ENCODING]]>

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
// CHECK-DAG: #[[ENCODING0:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
// CHECK-DAG: #[[ENCODING1:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [i8, i8, i32]>
// CHECK-DAG: #[[ENCODING2:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f16, f16, f32]>
// CHECK-DAG: #[[ENCODING3:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f16, f16, f16]>
// CHECK-DAG: #[[ENCODING4:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [bf16, bf16, f32]>
// CHECK-DAG: #[[ENCODING5:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [bf16, bf16, bf16]>
// CHECK:       func.func @set_encoding_with_batch_matmul_user
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]
// CHECK:         iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #[[ENCODING0]]>
// CHECK:         iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #[[ENCODING1]]>
// CHECK:         iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #[[ENCODING2]]>
// CHECK:         iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #[[ENCODING3]]>
// CHECK:         iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #[[ENCODING4]]>
// CHECK:         iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #[[ENCODING5]]>

// -----

#encoding = #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32]>
func.func @unset_encoding_fully_static(%arg0: tensor<3x5xf32, #encoding>) -> tensor<3x5xf32> {
  %0 = iree_encoding.unset_encoding %arg0 : tensor<3x5xf32, #encoding> -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}
// CHECK:       #[[ENCODING:.+]] = #iree_encoding.encoding<operand_index = 1 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
// CHECK:       func.func @unset_encoding_fully_static
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<3x5xf32, #[[ENCODING]]
// CHECK:         iree_encoding.unset_encoding %[[ARG0]] : tensor<3x5xf32, #[[ENCODING]]> -> tensor<3x5xf32>

// -----

#encoding = #iree_encoding.encoding<operand_index = 1 : i64, op_type = matmul, element_types = [f32, f32, f32]>
func.func @unset_encoding_fully_dynamic(%arg0: tensor<?x?xf32, #encoding>, %d0 : index, %d1 : index) -> tensor<?x?xf32> {
  %0 = iree_encoding.unset_encoding %arg0 : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>{%d0, %d1}
  return %0 : tensor<?x?xf32>
}
// CHECK:      #[[ENCODING:.+]] = #iree_encoding.encoding<operand_index = 1 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
// CHECK:      func.func @unset_encoding_fully_dynamic
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32, #[[ENCODING]]
// CHECK-SAME:      %[[D0:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[D1:[a-zA-Z0-9]+]]
// CHECK:        iree_encoding.unset_encoding %[[ARG0]] : tensor<?x?xf32, #[[ENCODING]]> -> tensor<?x?xf32>{%[[D0]], %[[D1]]}

// -----

#encoding = #iree_encoding.encoding<operand_index = 1 : i64, op_type = matmul, element_types = [f32, f32, f32]>
func.func @unset_encoding_ops_mixed_dynamic_static(%arg0: tensor<10x?xf32, #encoding>, %d0 : index) -> tensor<?x20xf32> {
  %0 = iree_encoding.unset_encoding %arg0 : tensor<10x?xf32, #encoding> -> tensor<?x20xf32>{%d0}
  return %0 : tensor<?x20xf32>
}
// CHECK:      #[[ENCODING:.+]] = #iree_encoding.encoding<operand_index = 1 : i64, op_type = matmul, element_types = [f32, f32, f32]>
// CHECK:      func.func @unset_encoding_ops_mixed_dynamic_static
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<10x?xf32, #[[ENCODING]]>
// CHECK-SAME:     %[[D0:[a-zA-Z0-9]+]]
// CHECK:        iree_encoding.unset_encoding %[[ARG0]] : tensor<10x?xf32, #[[ENCODING]]> -> tensor<?x20xf32>{%[[D0]]}

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
// CHECK-DAG:   #[[ENCODING0:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
// CHECK-DAG:   #[[ENCODING1:.+]] = #iree_encoding.encoding<operand_index = 1 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
// CHECK-DAG:   #[[ENCODING2:.+]] = #iree_encoding.encoding<operand_index = 2 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
// CHECK:       func.func @encoding_tensors_with_ops
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK-DAG:     %[[N:.+]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK:         %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:        tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING0]]>
// CHECK:         %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:        tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING1]]>
// CHECK:         %[[OUT:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:        tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING2]]>
// CHECK:         %[[GEMM:.+]] = linalg.matmul
// CHECK-SAME:        ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:        outs(%[[OUT]]
// CHECK:         %[[RESULT:.+]] = iree_encoding.unset_encoding %[[GEMM]] : tensor<?x?xf32, #[[ENCODING2]]> -> tensor<?x?xf32>{%[[M]], %[[N]]}
// CHECK:         return %[[RESULT]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], bcast_map = #map3>
func.func @set_encoding_ops_with_indexing_maps(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32, #encoding> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
  return %0 : tensor<?x?xf32, #encoding>
}
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:   #[[ENCODING:.+]] = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32],
// CHECK-SAME:    user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], bcast_map = #[[MAP3]]
// CHECK:       func.func @set_encoding_ops_with_indexing_maps(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]:
// CHECK:         iree_encoding.set_encoding %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32, #[[ENCODING]]>
