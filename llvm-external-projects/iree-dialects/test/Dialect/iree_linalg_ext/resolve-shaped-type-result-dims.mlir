// RUN: iree-dialects-opt -resolve-shaped-type-result-dims -split-input-file %s | FileCheck %s

func.func @pack_static(%arg0 : tensor<100x250xf32>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<100x250xf32> -> tensor<100x250xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>>
  %1 = tensor.dim %0, %c0 : tensor<100x250xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>>
  %2 = tensor.dim %0, %c1 : tensor<100x250xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>>
  return %1, %2 : index, index
}
// CHECK-LABEL: func @pack_static(
//   CHECK-DAG:   %[[C100:.+]] = arith.constant 100 : index
//   CHECK-DAG:   %[[C250:.+]] = arith.constant 250 : index
//       CHECK:   return %[[C100]], %[[C250]]

// -----

func.func @pack_dynamic(%arg0 : tensor<?x?xf32>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = iree_linalg_ext.set_encoding %arg0 : tensor<?x?xf32> -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>>
  %1 = tensor.dim %0, %c0 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>>
  %2 = tensor.dim %0, %c1 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL, role = LHS, elementTypes = [f32, f32, f32]>>
  return %1, %2 : index, index
}
//       CHECK: func @pack_dynamic(%[[ARG0:.+]]: tensor<?x?xf32>)
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK:   return %[[D0]], %[[D1]]
