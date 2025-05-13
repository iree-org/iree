// RUN: iree-opt --iree-flow-convert-to-flow --split-input-file --mlir-print-local-scope %s | FileCheck %s

func.func @mixed_concat(%arg0: tensor<2x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<4x?xf32>) -> tensor<?x?xf32> {
  %0 = tensor.concat dim(0) %arg0, %arg1, %arg2 : (tensor<2x?xf32>, tensor<?x?xf32>, tensor<4x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func @mixed_concat
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<2x?xf32>
//  CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-SAME:     %[[ARG2:.+]]: tensor<4x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[ARG0_D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[ARG1_D0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//   CHECK-DAG:   %[[ARG1_D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//       CHECK:   %[[OFFSET0:.+]] = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%[[ARG1_D0]]]
//       CHECK:   %[[ARG2_D1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//       CHECK:   %[[RESULT_D0:.+]] = affine.apply affine_map<()[s0] -> (s0 + 6)>()[%[[ARG1_D0]]]
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[RESULT_D0]], %[[ARG0_D1]])
//       CHECK:   %[[UPDATE0:.+]] = flow.tensor.update %[[ARG0]], %[[EMPTY]][%[[C0]], %[[C0]]]
//  CHECK-SAME:       : tensor<2x?xf32>{%[[ARG0_D1]]} -> %[[EMPTY]] as tensor<?x?xf32>{%[[RESULT_D0]], %[[ARG0_D1]]}
//       CHECK:   %[[UPDATE1:.+]] = flow.tensor.update %[[ARG1]], %[[UPDATE0]][%[[C2]], %[[C0]]]
//  CHECK-SAME:       : tensor<?x?xf32>{%[[ARG1_D0]], %[[ARG1_D1]]} -> %[[UPDATE0]] as tensor<?x?xf32>{%[[RESULT_D0]], %[[ARG0_D1]]}
//       CHECK:   %[[UPDATE2:.+]] = flow.tensor.update %[[ARG2]], %[[UPDATE1]][%[[OFFSET0]], %[[C0]]]
//  CHECK-SAME:       : tensor<4x?xf32>{%[[ARG2_D1]]} -> %[[UPDATE1]] as tensor<?x?xf32>{%[[RESULT_D0]], %[[ARG0_D1]]}

// -----

func.func @dont_lower_non_outer_dim_concat(%arg0: tensor<4x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<4x?xf32>) -> tensor<?x?xf32> {
  %0 = tensor.concat dim(1) %arg0, %arg1, %arg2 : (tensor<4x?xf32>, tensor<?x?xf32>, tensor<4x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func @dont_lower_non_outer_dim_concat
//       CHECK:   %[[CONCAT:.+]] = tensor.concat
//       CHECK:   return %[[CONCAT]]
// -----

func.func @mixed_concat_static_dim(%arg0: tensor<2x?xf32>, %arg1 : tensor<?x2xf32>) -> tensor<?x2xf32> {
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<2x?xf32>, tensor<?x2xf32>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
}
// CHECK-LABEL: func.func @mixed_concat_static_dim
//  CHECK-SAME:     %[[ARG0:.+]]: tensor<2x?xf32>
//  CHECK-SAME:     %[[ARG1:.+]]: tensor<?x2xf32>
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[ARG0_D1:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<2x?xf32>
//   CHECK-DAG:   %[[ARG1_D0:.+]] = tensor.dim %[[ARG1]], %[[C0]] : tensor<?x2xf32>
//       CHECK:   %[[RESULT_D0:.+]] = affine.apply affine_map<()[s0] -> (s0 + 2)>()[%[[ARG1_D0]]]
//       CHECK:   %[[EMPTY:.+]] = tensor.empty(%[[RESULT_D0]]) : tensor<?x2xf32>
//       CHECK:   %[[UPDATE1:.+]] = flow.tensor.update %[[ARG0]], %[[EMPTY]][%[[C0]], %[[C0]]]
//  CHECK-SAME:       : tensor<2x?xf32>{%[[ARG0_D1]]} -> %1 as tensor<?x2xf32>{%0}
//       CHECK:   %[[UPDATE2:.+]] = flow.tensor.update %[[ARG1]], %[[UPDATE1]][%[[C2]], %[[C0]]]
//  CHECK-SAME:       : tensor<?x2xf32>{%[[ARG1_D0]]} -> %2 as tensor<?x2xf32>{%0}
//       CHECK:   return %[[UPDATE2]] : tensor<?x2xf32>
