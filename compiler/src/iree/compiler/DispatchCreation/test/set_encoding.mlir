// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-set-encoding))" %s | FileCheck %s

util.func public @matmul_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  util.return %0 : tensor<100x500xf32>
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<100x250xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<250x500xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<100x500xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[MATMUL]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_f32f32f32_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_f32f32f32_dynamic(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>
//  CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<?x?xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<?x?xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[MATMUL]]{{.+}} -> tensor<?x?xf32>{%[[D0]], %[[D1]]}
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_i8i8i32(%arg0 : tensor<100x250xi8>, %arg1 : tensor<250x500xi8>,
    %arg2 : tensor<100x500xi32>) -> tensor<100x500xi32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xi8>, tensor<250x500xi8>)
      outs(%arg2 : tensor<100x500xi32>) -> tensor<100x500xi32>
  util.return %0 : tensor<100x500xi32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_i8i8i32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xi8>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xi8>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xi32>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<100x250xi8, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<250x500xi8, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<100x500xi32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[MATMUL]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_f16f16f32(%arg0 : tensor<100x250xf16>, %arg1 : tensor<250x500xf16>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250x500xf16>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  util.return %0 : tensor<100x500xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_f16f16f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<100x250xf16, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<250x500xf16, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<100x500xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[MATMUL]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_f16f16f16(%arg0 : tensor<100x250xf16>, %arg1 : tensor<250x500xf16>,
    %arg2 : tensor<100x500xf16>) -> tensor<100x500xf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250x500xf16>)
      outs(%arg2 : tensor<100x500xf16>) -> tensor<100x500xf16>
  util.return %0 : tensor<100x500xf16>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_f16f16f16(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf16>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<100x250xf16, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<250x500xf16, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<100x500xf16, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[MATMUL]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_bf16bf16f32(%arg0 : tensor<100x250xbf16>, %arg1 : tensor<250x500xbf16>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250x500xbf16>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  util.return %0 : tensor<100x500xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_bf16bf16f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xbf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xbf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<100x250xbf16, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<250x500xbf16, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<100x500xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[MATMUL]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_bf16bf16bf16(%arg0 : tensor<100x250xbf16>, %arg1 : tensor<250x500xbf16>,
    %arg2 : tensor<100x500xbf16>) -> tensor<100x500xbf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250x500xbf16>)
      outs(%arg2 : tensor<100x500xbf16>) -> tensor<100x500xbf16>
  util.return %0 : tensor<100x500xbf16>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_bf16bf16bf16(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xbf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xbf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xbf16>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<100x250xbf16, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<250x500xbf16, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<100x500xbf16, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[MATMUL]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_f32f32f32(%arg0 : tensor<64x100x250xf32>, %arg1 : tensor<64x250x500xf32>,
    %arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xf32>, tensor<64x250x500xf32>)
      outs(%arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32>
  util.return %0 : tensor<64x100x500xf32>
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xf32>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<64x100x250xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<64x250x500xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<64x100x500xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_f32f32f32_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?x?xf32>,
    %arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%arg2 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  util.return %0 : tensor<?x?x?xf32>
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_f32f32f32_dynamic(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>, %[[ARG1:.+]]: tensor<?x?x?xf32>, %[[ARG2:.+]]: tensor<?x?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//  CHECK-DAG:   %[[D2:.+]] = tensor.dim %[[ARG2]], %[[C2]]
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[BATCH_MATMUL]]{{.+}} -> tensor<?x?x?xf32>{%[[D0]], %[[D1]], %[[D2]]}
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_f16f16f16(%arg0 : tensor<64x100x250xf16>, %arg1 : tensor<64x250x500xf16>,
    %arg2 : tensor<64x100x500xf16>) -> tensor<64x100x500xf16> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xf16>, tensor<64x250x500xf16>)
      outs(%arg2 : tensor<64x100x500xf16>) -> tensor<64x100x500xf16>
  util.return %0 : tensor<64x100x500xf16>
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_f16f16f16(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xf16>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<64x100x250xf16, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<64x250x500xf16, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<64x100x500xf16, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f16, f16, f16], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_f16f16f32(%arg0 : tensor<64x100x250xf16>, %arg1 : tensor<64x250x500xf16>,
    %arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xf16>, tensor<64x250x500xf16>)
      outs(%arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32>
  util.return %0 : tensor<64x100x500xf32>
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_f16f16f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xf32>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<64x100x250xf16, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<64x250x500xf16, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<64x100x500xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f16, f16, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_bf16bf16bf16(%arg0 : tensor<64x100x250xbf16>, %arg1 : tensor<64x250x500xbf16>,
    %arg2 : tensor<64x100x500xbf16>) -> tensor<64x100x500xbf16> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xbf16>, tensor<64x250x500xbf16>)
      outs(%arg2 : tensor<64x100x500xbf16>) -> tensor<64x100x500xbf16>
  util.return %0 : tensor<64x100x500xbf16>
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_bf16bf16bf16(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xbf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xbf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xbf16>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<64x100x250xbf16, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<64x250x500xbf16, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<64x100x500xbf16, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [bf16, bf16, bf16], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_bf16bf16f32(%arg0 : tensor<64x100x250xbf16>, %arg1 : tensor<64x250x500xbf16>,
    %arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xbf16>, tensor<64x250x500xbf16>)
      outs(%arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32>
  util.return %0 : tensor<64x100x500xf32>
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_bf16bf16f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xbf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xbf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xf32>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<64x100x250xbf16, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<64x250x500xbf16, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<64x100x500xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [bf16, bf16, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_i8i8i32(%arg0 : tensor<64x100x250xi8>, %arg1 : tensor<64x250x500xi8>,
    %arg2 : tensor<64x100x500xi32>) -> tensor<64x100x500xi32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x100x250xi8>, tensor<64x250x500xi8>)
      outs(%arg2 : tensor<64x100x500xi32>) -> tensor<64x100x500xi32>
  util.return %0 : tensor<64x100x500xi32>
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_i8i8i32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<64x100x250xi8>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<64x250x500xi8>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<64x100x500xi32>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<64x100x250xi8, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<64x250x500xi8, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<64x100x500xi32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [i8, i8, i32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @vecmat_f32f32f32(%arg0 : tensor<250xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<500xf32>) -> tensor<500xf32> {
  %0 = linalg.vecmat ins(%arg0, %arg1 : tensor<250xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<500xf32>) -> tensor<500xf32>
  util.return %0 : tensor<500xf32>
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d1, d0)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0)>
//      CHECK: util.func public @vecmat_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<500xf32>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<250xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 1, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<250x500xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 1, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<500xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 1, 32, 32>>>
//      CHECK:   %[[VECMAT:.+]] = linalg.vecmat
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[VECMAT]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @matvec_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250xf32>,
    %arg2 : tensor<100xf32>) -> tensor<100xf32> {
  %0 = linalg.matvec ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250xf32>)
      outs(%arg2 : tensor<100xf32>) -> tensor<100xf32>
  util.return %0 : tensor<100xf32>
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0)>
//      CHECK: util.func public @matvec_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100xf32>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<100x250xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 1, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<250xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 1, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<100xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 1, 32>>>
//      CHECK:   %[[MATVEC:.+]] = linalg.matvec
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[MATVEC]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_vecmat_f32f32f32(%arg0 : tensor<3x250xf32>, %arg1 : tensor<3x250x500xf32>,
    %arg2 : tensor<3x500xf32>) -> tensor<3x500xf32> {
  %0 = linalg.batch_vecmat ins(%arg0, %arg1 : tensor<3x250xf32>, tensor<3x250x500xf32>)
      outs(%arg2 : tensor<3x500xf32>) -> tensor<3x500xf32>
  util.return %0 : tensor<3x500xf32>
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @batch_vecmat_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<3x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<3x250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<3x500xf32>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<3x250xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 1, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<3x250x500xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 1, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<3x500xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 1, 32, 32>>>
//      CHECK:   %[[VECMAT:.+]] = linalg.batch_vecmat
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[VECMAT]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matvec_f32f32f32_dynamic(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.batch_matvec ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %0 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @batch_matvec_f32f32f32_dynamic(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<?x?x?xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 1, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<?x?xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 1, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<?x?xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 1, 32>>>
//      CHECK:   %[[BATCH_MATVEC:.+]] = linalg.batch_matvec
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[BATCH_MATVEC]]{{.+}} -> tensor<?x?xf32>{%[[D0]], %[[D1]]}
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @fold_fill_with_set_encoding(%arg0 : index, %arg1 : index)
  -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], round_dims_to = array<i64: 32, 32, 32>>> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = iree_encoding.set_encoding %1 : tensor<?x?xf32>
      -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], round_dims_to = array<i64: 32, 32, 32>>>
  util.return %2 : tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0, op_type = matmul, element_types = [f32, f32, f32], round_dims_to = array<i64: 32, 32, 32>>>
}
//      CHECK: util.func public @fold_fill_with_set_encoding(
//      CHECK:   %[[EMPTY:.+]] = tensor.empty(%{{.+}}, %{{.+}}) : tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] : tensor<?x?xf32, #iree_encoding.encoding<operand_index = 0 : i64, op_type = matmul, element_types = [f32, f32, f32], round_dims_to = array<i64: 32, 32, 32>>>)
//      CHECK:   util.return %[[FILL]]

// -----

util.func public @fold_fill_with_tensor_pad(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index)
    -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], round_dims_to = array<i64: 32, 32, 32>>> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = tensor.pad %1 low[0, 0] high[%arg2, %arg3] {
  ^bb0(%b0: index, %b1 : index):
    tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<?x?xf32>
      -> tensor<?x?xf32, #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], round_dims_to = array<i64: 32, 32, 32>>>
  util.return %3 : tensor<?x?xf32, #iree_encoding.encoding<operand_index = 2, op_type = matmul, element_types = [f32, f32, f32], round_dims_to = array<i64: 32, 32, 32>>>
}
//      CHECK: util.func public @fold_fill_with_tensor_pad(
//      CHECK:   %[[EMPTY:.+]] = tensor.empty(
// CHECK-SAME:       tensor<?x?xf32, #iree_encoding.encoding<operand_index = 2 : i64, op_type = matmul, element_types = [f32, f32, f32], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   util.return %[[FILL]]

// -----

#compilation0 = #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0]]>,
    translation_info  = <CPUDefault>>

#compilation1 = #iree_codegen.compilation_info<
    lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 0, 0, 0]]>,
    translation_info  = <CPUDefault>>


util.func public @preset_compilation_info(
    %arg0 : tensor<?x?xf32>,
    %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>,
    %arg3 : tensor<?x?x?xf32>,
    %arg4 : tensor<?x?x?xf32>,
    %arg5 : tensor<?x?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?x?xf32>) {
  %0 = linalg.matmul {compilation_info = #compilation0} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.batch_matmul {compilation_info = #compilation1} ins(%arg3, %arg4 : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
      outs(%arg5 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  util.return %0, %1 : tensor<?x?xf32>, tensor<?x?x?xf32>
}
// CHECK-LABEL: util.func public @preset_compilation_info
// CHECK-NOT:     set_encoding
// CHECK-NOT:     unset_encoding
// CHECK:         linalg.matmul
// CHECK:         linalg.batch_matmul

// -----

util.func public @batch_matmul_truncf_f16f16f32(%arg0 : tensor<64x100x250xf32>, %arg1 : tensor<64x250x500xf32>,
      %arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32> {
  %0 = tensor.empty() : tensor<64x250x500xf16>
  %casted0 = arith.truncf %arg0 : tensor<64x100x250xf32> to tensor<64x100x250xf16>
  %casted1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                                              affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                              iterator_types = ["parallel", "parallel", "parallel"]}
                              ins(%arg1 : tensor<64x250x500xf32>)
                              outs(%0 : tensor<64x250x500xf16>) {
  ^bb0(%in: f32, %out: f16):
      %2 = arith.truncf %in : f32 to f16
      linalg.yield %2 : f16
  } -> tensor<64x250x500xf16>
  %1 = linalg.batch_matmul ins(%casted0, %casted1 : tensor<64x100x250xf16>, tensor<64x250x500xf16>)
      outs(%arg2 : tensor<64x100x500xf32>) -> tensor<64x100x500xf32>
  util.return %1 : tensor<64x100x500xf32>
}

//      CHECK: util.func public @batch_matmul_truncf_f16f16f32(%[[ARG0:.+]]: tensor<64x100x250xf32>, %[[ARG1:.+]]: tensor<64x250x500xf32>
//  CHECK-DAG: %[[INIT:.+]] = tensor.empty() : tensor<64x250x500xf16>
//  CHECK-DAG: arith.truncf %[[ARG0]] : tensor<64x100x250xf32> to tensor<64x100x250xf16>
//      CHECK: linalg.generic
// CHECK-SAME:   ins(%[[ARG1]] : tensor<64x250x500xf32>)
// CHECK-SAME:   outs(%[[INIT]] : tensor<64x250x500xf16>)
//      CHECK: element_types = [f16, f16, f32]

// -----

util.func public @matmul_casted_from_i1_f32f32f32(%arg0 : tensor<64x256xi1>,
    %arg1 : tensor<256x128xf32>) -> tensor<64x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %casted = arith.uitofp %arg0 : tensor<64x256xi1> to tensor<64x256xf32>
  %0 = tensor.empty() : tensor<64x128xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = linalg.matmul ins(%casted, %arg1 : tensor<64x256xf32>, tensor<256x128xf32>) outs(%1 : tensor<64x128xf32>) -> tensor<64x128xf32>
  util.return %2 : tensor<64x128xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_casted_from_i1_f32f32f32
// CHECK:         set_encoding {{.+}} tensor<64x256xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
// CHECK:         set_encoding {{.+}} tensor<256x128xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
// CHECK:         set_encoding {{.+}} tensor<64x128xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>

// -----

util.func public @matmul_generic_casted_from_i1_f32f32f32(%arg0 : tensor<64x256xi1>,
    %arg1 : tensor<256x128xf32>) -> tensor<64x128xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<64x256xf32>
  %casted = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                              affine_map<(d0, d1) -> (d0, d1)>],
                              iterator_types = ["parallel", "parallel"]}
                              ins(%arg0 : tensor<64x256xi1>)
                              outs(%init : tensor<64x256xf32>) {
  ^bb0(%in: i1, %out: f32):
      %1 = arith.uitofp %in : i1 to f32
      linalg.yield %1 : f32
  } -> tensor<64x256xf32>
  %0 = tensor.empty() : tensor<64x128xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64x128xf32>) -> tensor<64x128xf32>
  %2 = linalg.matmul ins(%casted, %arg1 : tensor<64x256xf32>, tensor<256x128xf32>) outs(%1 : tensor<64x128xf32>) -> tensor<64x128xf32>
  util.return %2 : tensor<64x128xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_generic_casted_from_i1_f32f32f32
// CHECK:         set_encoding {{.+}} tensor<64x256xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
// CHECK:         set_encoding {{.+}} tensor<256x128xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>
// CHECK:         set_encoding {{.+}} tensor<64x128xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 32, 32>>>

// -----

util.func public @matmul_f32f32f32_narrow_M(%arg0 : tensor<2x250xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<2x500xf32>) -> tensor<2x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<2x250xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<2x500xf32>) -> tensor<2x500xf32>
  util.return %0 : tensor<2x500xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_f32f32f32_narrow_M(
//      CHECK:  iree_encoding.set_encoding
// CHECK-SAME:    tensor<2x250xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 2, 32, 32>>>
//      CHECK:  iree_encoding.set_encoding
// CHECK-SAME:    tensor<250x500xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 2, 32, 32>>>
//      CHECK:  iree_encoding.set_encoding
// CHECK-SAME:    tensor<2x500xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 2, 32, 32>>>
//      CHECK:   linalg.matmul

// -----

util.func public @batch_matmul_f32f32f32_narrow_MN(%arg0 : tensor<64x4x250xf32>, %arg1 : tensor<64x250x2xf32>,
    %arg2 : tensor<64x4x2xf32>) -> tensor<64x4x2xf32> {
  %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<64x4x250xf32>, tensor<64x250x2xf32>)
      outs(%arg2 : tensor<64x4x2xf32>) -> tensor<64x4x2xf32>
  util.return %0 : tensor<64x4x2xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_f32f32f32_narrow_MN(
//      CHECK:   iree_encoding.set_encoding
// CHECK-SAME:     tensor<64x4x250xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 2, 32>>>
//      CHECK:   iree_encoding.set_encoding
// CHECK-SAME:     tensor<64x250x2xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 2, 32>>>
//      CHECK:   iree_encoding.set_encoding
// CHECK-SAME:     tensor<64x4x2xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]], round_dims_to = array<i64: 32, 2, 32>>>
//      CHECK:   linalg.batch_matmul

// -----

util.func public @matmul_transpose_a_f32f32f32(%arg0 : tensor<250x100xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul_transpose_a ins(%arg0, %arg1 : tensor<250x100xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  util.return %0 : tensor<100x500xf32>
}

//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d0)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_transpose_a_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<250x100xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<250x100xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<250x500xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<100x500xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul_transpose_a
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[MATMUL]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @matmul_transpose_b_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<500x250xf32>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<500x250xf32>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  util.return %0 : tensor<100x500xf32>
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @matmul_transpose_b_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<500x250xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<100x250xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<500x250xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<100x500xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul_transpose_b
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[MATMUL]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_transpose_a_f32f32f32(%arg0 : tensor<2x250x100xf32>, %arg1 : tensor<2x250x500xf32>,
    %arg2 : tensor<2x100x500xf32>) -> tensor<2x100x500xf32> {
  %0 = linalg.batch_matmul_transpose_a ins(%arg0, %arg1 : tensor<2x250x100xf32>, tensor<2x250x500xf32>)
      outs(%arg2 : tensor<2x100x500xf32>) -> tensor<2x100x500xf32>
  util.return %0 : tensor<2x100x500xf32>
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_transpose_a_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<2x250x100xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<2x250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<2x100x500xf32>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<2x250x100xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<2x250x500xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<2x100x500xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul_transpose_a
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @batch_matmul_transpose_b_f32f32f32(%arg0 : tensor<2x100x250xf32>, %arg1 : tensor<2x500x250xf32>,
    %arg2 : tensor<2x100x500xf32>) -> tensor<2x100x500xf32> {
  %0 = linalg.batch_matmul_transpose_b ins(%arg0, %arg1 : tensor<2x100x250xf32>, tensor<2x500x250xf32>)
      outs(%arg2 : tensor<2x100x500xf32>) -> tensor<2x100x500xf32>
  util.return %0 : tensor<2x100x500xf32>
}
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
//      CHECK: util.func public @batch_matmul_transpose_b_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<2x100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<2x500x250xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<2x100x500xf32>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<2x100x250xf32, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<2x500x250xf32, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<2x100x500xf32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f32, f32, f32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 32, 32, 32>>>
//      CHECK:   %[[BATCH_MATMUL:.+]] = linalg.batch_matmul_transpose_b
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[BATCH_MATMUL]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @generic_batch_vecmat_transposed_i16u4i32(%arg0 : tensor<32x128xi16>, %arg1 : tensor<4096x32x128xi4>,
    %arg2 : tensor<4096x32xi32>) -> tensor<4096x32xi32> {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<32x128xi16>, tensor<4096x32x128xi4>) outs(%arg2 : tensor<4096x32xi32>) {
  ^bb0(%in: i16, %in_5: i4, %out: i32):
    %22 = arith.extsi %in : i16 to i32
    %23 = arith.extui %in_5 : i4 to i32
    %24 = arith.muli %22, %23 : i32
    %25 = arith.addi %24, %out : i32
    linalg.yield %25 : i32
  } -> tensor<4096x32xi32>
  util.return %0 : tensor<4096x32xi32>
}

//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//      CHECK: util.func public @generic_batch_vecmat_transposed_i16u4i32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<32x128xi16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<4096x32x128xi4>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<4096x32xi32>
//      CHECK:   %[[LHS:.+]] = iree_encoding.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<32x128xi16, #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [i16, ui4, i32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 1, 32, 32>>>
//      CHECK:   %[[RHS:.+]] = iree_encoding.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<4096x32x128xi4, #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [i16, ui4, i32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 1, 32, 32>>>
//      CHECK:   %[[OUTS:.+]] = iree_encoding.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<4096x32xi32> -> tensor<4096x32xi32, #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [i16, ui4, i32], user_indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], round_dims_to = array<i64: 1, 32, 32>>>
//      CHECK:   %[[GENERIC:.+]] = linalg.generic
// CHECK-SAME:       indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]]
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_encoding.unset_encoding %[[GENERIC]]
//      CHECK:   util.return %[[RESULT]]

// -----

util.func public @dot(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<f32> {
  %res = "stablehlo.dot"(%arg0, %arg1) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<f32>
  util.return %res : tensor<f32>
}

// CHECK: util.func public @dot(
// CHECK: stablehlo.dot %{{.*}}, %{{.*}} : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<f32>

// -----

util.func public @multi_m_dim_generic(%arg0 : tensor<64x4x128xf32>, %arg1 : tensor<128x512xf32>,
    %arg2 : tensor<64x4x512xf32>) -> tensor<64x4x512xf32> {
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
                         affine_map<(d0, d1, d2, d3) -> (d2, d1)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%arg0, %arg1 : tensor<64x4x128xf32>, tensor<128x512xf32>) outs(%arg2 : tensor<64x4x512xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<64x4x512xf32>
  util.return %4 : tensor<64x4x512xf32>
}

//      CHECK: util.func public @multi_m_dim_generic(
//      CHECK:   linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<64x4x128xf32>, tensor<128x512xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<64x4x512xf32>)

// -----

util.func public @multi_n_dim_generic(%arg0 : tensor<256x128xf32>, %arg1 : tensor<128x64x8xf32>,
    %arg2 : tensor<256x64x8xf32>) -> tensor<256x64x8xf32> {
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2)>,
                         affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
        ins(%arg0, %arg1 : tensor<256x128xf32>, tensor<128x64x8xf32>) outs(%arg2 : tensor<256x64x8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<256x64x8xf32>
  util.return %4 : tensor<256x64x8xf32>
}

//      CHECK: util.func public @multi_n_dim_generic(
//      CHECK:   linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<256x128xf32>, tensor<128x64x8xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<256x64x8xf32>)

// -----

util.func public @multi_k_dim_generic(%arg0 : tensor<256x64x2xf32>, %arg1 : tensor<64x2x512xf32>,
    %arg2 : tensor<256x512xf32>) -> tensor<256x512xf32> {
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>,
                         affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>,
                         affine_map<(d0, d1, d2, d3) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
        ins(%arg0, %arg1 : tensor<256x64x2xf32>, tensor<64x2x512xf32>) outs(%arg2 : tensor<256x512xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<256x512xf32>
  util.return %4 : tensor<256x512xf32>
}

//      CHECK: util.func public @multi_k_dim_generic(
//      CHECK:   linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<256x64x2xf32>, tensor<64x2x512xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<256x512xf32>)

// -----

util.func public @multi_batch_dim_generic(%arg0 : tensor<4x8x256x128xf32>, %arg1 : tensor<4x8x128x512xf32>,
    %arg2 : tensor<4x8x256x512xf32>) -> tensor<4x8x256x512xf32> {
    %4 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>,
                         affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>],
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%arg0, %arg1 : tensor<4x8x256x128xf32>, tensor<4x8x128x512xf32>) outs(%arg2 : tensor<4x8x256x512xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %5, %out : f32
      linalg.yield %6 : f32
    } -> tensor<4x8x256x512xf32>
  util.return %4 : tensor<4x8x256x512xf32>
}

//      CHECK: util.func public @multi_batch_dim_generic(
//      CHECK:   linalg.generic
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<4x8x256x128xf32>, tensor<4x8x128x512xf32>)
// CHECK-SAME:      outs(%{{.*}} : tensor<4x8x256x512xf32>)

// -----

#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
util.func public @broadcasting_dequant_op(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> tensor<?x?x?xi32> {
  %c0_i32 = arith.constant 0 : i32
  %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
  %2 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<?x?xi8>{%0, %1}
  %3 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
  %4 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
  %5 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[2] : index
  %6 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<?x?x?xi32>{%3, %4, %5}
  %7 = flow.dispatch.region -> (tensor<?x?x?xi32>{%3, %0, %4}) {
    %9 = tensor.empty(%3, %0, %4) : tensor<?x?x?xi32>
    %c0_i32_0 = arith.constant 0 : i32
    %10 = tensor.empty(%3, %0, %1) : tensor<?x?x?xi32>
    %11 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<?x?xi8>) outs(%10 : tensor<?x?x?xi32>) {
    ^bb0(%in: i8, %out: i32):
      %14 = arith.extui %in : i8 to i32
      linalg.yield %14 : i32
    } -> tensor<?x?x?xi32>
    %12 = linalg.fill ins(%c0_i32_0 : i32) outs(%9 : tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
    %13 = linalg.batch_matmul_transpose_b ins(%11, %6 : tensor<?x?x?xi32>, tensor<?x?x?xi32>) outs(%12 : tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
    flow.return %13 : tensor<?x?x?xi32>
  }
  util.return %7 : tensor<?x?x?xi32>
}
// CHECK-DAG:  #[[MAP:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK:      util.func public @broadcasting_dequant_op(
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-DAG:  %[[ARG0_D0:.+]] = hal.buffer_view.dim<%[[ARG0]] : !hal.buffer_view>[0] : index
// CHECK-DAG:  %[[ARG0_D1:.+]] = hal.buffer_view.dim<%[[ARG0]] : !hal.buffer_view>[1] : index
// CHECK-DAG:  %[[ARG1_D0:.+]] = hal.buffer_view.dim<%[[ARG1]] : !hal.buffer_view>[0] : index
// CHECK-DAG:  %[[ARG1_D1:.+]] = hal.buffer_view.dim<%[[ARG1]] : !hal.buffer_view>[1] : index
// CHECK-DAG:  %[[ARG1_D2:.+]] = hal.buffer_view.dim<%[[ARG1]] : !hal.buffer_view>[2] : index
// CHECK:      %{{.+}} = flow.dispatch.region
// CHECK:        %[[BCAST:.+]] = linalg.generic
// CHECK:        %[[LHS:.+]] = iree_encoding.set_encoding %[[BCAST]] : tensor<?x?x?xi32>
// CHECK-SAME:    -> tensor<?x?x?xi32, #iree_encoding.encoding
// CHECK-SAME:      operand_index = 0 : index
// CHECK-SAME:      element_types = [i32, i32, i32]
// CHECK-SAME:      user_indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]]
// CHECK-SAME:      round_dims_to = array<i64: 32, 32, 32>
// CHECK:        %[[RHS:.+]] = iree_encoding.set_encoding %{{.+}} : tensor<?x?x?xi32>
// CHECK-SAME:      operand_index = 1 : index
// CHECK-SAME:      element_types = [i32, i32, i32]
// CHECK-SAME:      user_indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]]
// CHECK-SAME:      round_dims_to = array<i64: 32, 32, 32>
// CHECK:        %[[INIT:.+]] = tensor.empty({{.+}}) : tensor<?x?x?xi32, #iree_encoding.encoding
// CHECK-SAME:      operand_index = 2 : index
// CHECK-SAME:      element_types = [i32, i32, i32]
// CHECK-SAME:      user_indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]]
// CHECK-SAME:      round_dims_to = array<i64: 32, 32, 32>
// CHECK:        %[[FILL:.+]] = linalg.fill ins({{.+}}) outs(%[[INIT]]
// CHECK:        %[[GEMM:.+]] = linalg.batch_matmul_transpose_b
// CHECK-SAME:     ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:    outs(%[[FILL]]
// CHECK:        %[[UNSET:.+]] = iree_encoding.unset_encoding %[[GEMM]]{{.+}} -> tensor<?x?x?xi32>{%[[ARG1_D0]], %[[ARG0_D0]], %[[ARG1_D1]]}
// CHECK:        flow.return %[[UNSET]]
