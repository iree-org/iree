// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK: @set_encoding_ops(%[[ARG0:.+]]: tensor<?x?xf32>)
func.func @set_encoding_ops(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>> {
  // CHECK: iree_encoding.set_encoding %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32]>>
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>>
  return %0 : tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>>
}

// -----

// CHECK: @set_encoding_ops_mixed_dynamic_static(%[[ARG0:.+]]: tensor<?x10xf32>)
func.func @set_encoding_ops_mixed_dynamic_static(%arg0: tensor<?x10xf32>) -> tensor<20x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>> {
  // CHECK: iree_encoding.set_encoding %[[ARG0]] : tensor<?x10xf32> -> tensor<20x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32]>>
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x10xf32> -> tensor<20x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>>
  return %0 : tensor<20x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>>
}

// -----

// CHECK: @set_encoding_with_batch_matmul_user(%[[ARG0:.+]]: tensor<?x?x?xf32>)
func.func @set_encoding_with_batch_matmul_user(%arg0: tensor<?x?x?xf32>) {
  // CHECK: iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32]>>
  iree_encoding.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>>
  // CHECK: iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [i8, i8, i32]>>
  iree_encoding.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [i8, i8, i32]>>
  // CHECK: iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f16, f16, f32]>>
  iree_encoding.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f32]>>
  // CHECK: iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f16, f16, f16]>>
  iree_encoding.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f16, f16, f16]>>
  // CHECK: iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [bf16, bf16, f32]>>
  iree_encoding.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, f32]>>
  // CHECK: iree_encoding.set_encoding %[[ARG0]] : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [bf16, bf16, bf16]>>
  iree_encoding.set_encoding %arg0 : tensor<?x?x?xf32> -> tensor<?x?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [bf16, bf16, bf16]>>
  return
}

// -----

// CHECK: @unset_encoding_ops(%[[ARG0:.+]]: tensor<?x?xf32, #iree_encoding.encoding<operand_index = 1 : i64, op_type = matmul, element_types = [f32, f32, f32]>>)
func.func @unset_encoding_ops(%arg0: tensor<?x?xf32, #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32]>>) -> tensor<?x?xf32> {
  // CHECK: iree_encoding.unset_encoding %[[ARG0]] : tensor<?x?xf32, #iree_encoding.encoding<operand_index = 1 : i64, op_type = matmul, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
  %0 = iree_encoding.unset_encoding %arg0 : tensor<?x?xf32, #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK: @unset_encoding_ops_mixed_dynamic_static(%[[ARG0:.+]]: tensor<10x?xf32, #iree_encoding.encoding<operand_index = 1 : i64, op_type = matmul, element_types = [f32, f32, f32]>>)
func.func @unset_encoding_ops_mixed_dynamic_static(%arg0: tensor<10x?xf32, #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32]>>) -> tensor<?x20xf32> {
  // CHECK: iree_encoding.unset_encoding %[[ARG0]] : tensor<10x?xf32, #iree_encoding.encoding<operand_index = 1 : i64, op_type = matmul, element_types = [f32, f32, f32]>>
  %0 = iree_encoding.unset_encoding %arg0 : tensor<10x?xf32, #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32]>> -> tensor<?x20xf32>
  return %0 : tensor<?x20xf32>
}

// -----

func.func @encoding_tensors_with_ops(%arg0 : tensor<?x?xf32>,
    %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>>
  %1 = iree_encoding.set_encoding %arg1 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32]>>
  %2 = iree_encoding.set_encoding %arg2 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32]>>
  %3 = linalg.matmul
      ins(%0, %1 : tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32]>>, tensor<?x?xf32, #iree_encoding.encoding<operand_index = 1, op_type = matmul, element_types = [f32, f32, f32]>>)
      outs(%2 : tensor<?x?xf32, #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32]>>)
      -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32]>>
  %4 = iree_encoding.unset_encoding %3 : tensor<?x?xf32, #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32]>> -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}
// CHECK-LABEL: func.func @encoding_tensors_with_ops
//  CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//       CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
//  CHECK-SAME:       tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32]>>
//       CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
//  CHECK-SAME:       tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 1 : i64, op_type = matmul, element_types = [f32, f32, f32]>>
//       CHECK:   %[[OUT:.+]] = iree_encoding.set_encoding %[[ARG2]]
//  CHECK-SAME:       tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 2 : i64, op_type = matmul, element_types = [f32, f32, f32]>>
//       CHECK:   %[[GEMM:.+]] = linalg.matmul
//  CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
//  CHECK-SAME:       outs(%[[OUT]] :
//       CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[GEMM]]
//       CHECK:   return %[[RESULT]]

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @set_encoding_ops_with_indexing_maps(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], bcast_map = #map3>> {
  %0 = iree_encoding.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], bcast_map = #map3>>
  return %0 : tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], bcast_map = #map3>>
}
// CHECK-DAG:   #[[MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG:   #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK:       func.func @set_encoding_ops_with_indexing_maps(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]:
// CHECK:         iree_encoding.set_encoding %[[ARG0]] : tensor<?x?xf32> -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]], bcast_map = #[[MAP3]]>>
