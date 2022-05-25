// RUN: iree-opt --split-input-file --verify-diagnostics --iree-flow- --pass-pipeline="func.func(iree-flow-dispatch-linalg-on-tensors-pass), cse, canonicalize, cse" %s | FileCheck %s

func.func @pad_op_consumer_fusion(%arg0 : tensor<?x?xf32>, %low0 : index,
  %low1: index, %high0 : index, %high1 : index, %arg1 : tensor<?x?xf32>,
  %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.pad %arg0 low[%low0, %low1] high[%high0, %high1] {
    ^bb0(%b0 : index, %b1 : index):
      tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  %1 = linalg.matmul ins(%0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}
//      CHECK: func.func @pad_op_consumer_fusion
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//      CHECK:   %[[DISPATCH:.+]] = flow.dispatch.workgroups
// CHECK-SAME:       (%[[ARG0]]
// CHECK-NEXT:     %[[ARG7:[a-zA-Z0-9]+]]: !flow.dispatch.tensor<readonly:?x?xf32>
//      CHECK:     %[[PAD:.+]] = tensor.pad %[[ARG7]]
//      CHECK:     linalg.matmul ins(%[[PAD]]
//      CHECK:   return %[[DISPATCH]]
