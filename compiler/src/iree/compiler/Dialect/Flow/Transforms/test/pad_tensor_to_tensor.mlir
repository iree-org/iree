// RUN: iree-opt --split-input-file --iree-flow-pad-tensor-to-subtensor-insert --canonicalize %s | FileCheck %s

module  {
  func.func @pad_tensor(%arg0 : tensor<?x?xf32>, %arg1 : tensor<f32>, %arg2 : index, %arg3 : index) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %0 = tensor.extract %arg1[] : tensor<f32>
    %1 = tensor.pad %arg0 low[%c4, %arg2] high[%arg3, %c3]  {
    ^bb0(%arg4: index, %arg5: index):  // no predecessors
      tensor.yield %0 : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
    return %1 : tensor<?x?xf32>
  }
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 4)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 3)>
//       CHECK: func.func @pad_tensor
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<f32>
//  CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]+]]: index
//  CHECK-SAME:   %[[ARG3:[a-zA-Z0-9_]+]]: index
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1
//   CHECK-DAG:   %[[VAL:.+]] = tensor.extract %[[ARG1]]
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//   CHECK-DAG:   %[[RD0:.+]] = affine.apply #[[MAP0]]()[%[[ARG3]], %[[D0]]]
//   CHECK-DAG:   %[[RD1:.+]] = affine.apply #[[MAP1]]()[%[[ARG2]], %[[D1]]]
//       CHECK:   %[[INIT:.+]] = linalg.init_tensor [%[[RD0]], %[[RD1]]]
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       ins(%[[VAL]] :
//  CHECK-SAME:       outs(%[[INIT]] :
//       CHECK:   %[[RESULT:.+]] = tensor.insert_slice %[[ARG0]] into %[[FILL]][4, %[[ARG2]]] [%[[D0]], %[[D1]]] [1, 1]
//       CHECK:   return %[[RESULT]]

// -----

module  {
  func.func @pad_tensor_static(%arg0: tensor<12x4xf32>, %arg1: tensor<f32>) -> tensor<18x12xf32> {
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c5 = arith.constant 5 : index
    %c3 = arith.constant 3 : index
    %0 = tensor.extract %arg1[] : tensor<f32>
    %1 = tensor.pad %arg0 low[%c4, %c5] high[%c2, %c3]  {
    ^bb0(%arg2: index, %arg3: index):  // no predecessors
      tensor.yield %0 : f32
    } : tensor<12x4xf32> to tensor<18x12xf32>
    return %1 : tensor<18x12xf32>
  }
}
// CHECK-LABEL: func.func @pad_tensor_static
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: tensor<12x4xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: tensor<f32>
//   CHECK-DAG:   %[[VAL:.+]] = tensor.extract %[[ARG1]]
//       CHECK:   %[[INIT:.+]] = linalg.init_tensor [18, 12]
//       CHECK:   %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:       ins(%[[VAL]] :
//  CHECK-SAME:       outs(%[[INIT]] :
//       CHECK:   %[[RESULT:.+]] = tensor.insert_slice %[[ARG0]] into %[[FILL]][4, 5] [12, 4] [1, 1]
//       CHECK:   return %[[RESULT]]
