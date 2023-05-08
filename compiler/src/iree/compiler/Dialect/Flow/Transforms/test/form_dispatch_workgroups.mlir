// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-flow-form-dispatch-workgroups))" --split-input-file %s | FileCheck %s

func.func @existing_count_region(%arg0 : index, %arg1 : index) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %0 = flow.dispatch.region[%arg0, %arg1] -> (tensor<?x?xf32>{%arg0, %arg1}) {
    %1 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
    flow.return %1 : tensor<?x?xf32>
  } count(%arg2 : index, %arg3 : index) -> (index, index, index) {
    flow.return %arg2, %arg3, %c1 : index, index, index
  }
  return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func @existing_count_region(
//       CHECK:   count(%[[ARG2:[a-zA-Z0-9]+]]: index, %[[ARG3:[a-zA-Z0-9]+]]: index)
//       CHECK:     %[[C1:.+]] = arith.constant 1 : index
//       CHECK:     flow.return %[[ARG2]], %[[ARG3]], %[[C1]]
