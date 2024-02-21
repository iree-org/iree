// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-flow-form-dispatch-workgroups))" --split-input-file %s | FileCheck %s

util.func public @existing_count_region(%arg0 : index, %arg1 : index) -> tensor<?x?xf32> {
  %c1 = arith.constant 1 : index
  %0 = flow.dispatch.region[%arg0, %arg1] -> (tensor<?x?xf32>{%arg0, %arg1}) {
    %1 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
    flow.return %1 : tensor<?x?xf32>
  } count(%arg2 : index, %arg3 : index) -> (index, index, index) {
    flow.return %arg2, %arg3, %c1 : index, index, index
  }
  util.return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @existing_count_region(
//       CHECK:   count(%[[ARG2:[a-zA-Z0-9]+]]: index, %[[ARG3:[a-zA-Z0-9]+]]: index)
//       CHECK:     %[[C1:.+]] = arith.constant 1 : index
//       CHECK:     flow.return %[[ARG2]], %[[ARG3]], %[[C1]]

// -----

util.func public @simple_test_with_cfg(%arg0: i1) -> (tensor<10x20xf32>) {
  %cst = arith.constant dense<1.000000e+00> : tensor<10x20xf32>
  %0 = flow.dispatch.region -> (tensor<10x20xf32>) {
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<10x20xf32>
    cf.cond_br %arg0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %2 = tensor.empty() : tensor<10x20xf32>
    flow.return %2 : tensor<10x20xf32>
  ^bb2:  // pred: ^bb0
    flow.return %cst_0 : tensor<10x20xf32>
  }
  util.return %0 : tensor<10x20xf32>
}
// CHECK-LABEL: util.func public @simple_test_with_cfg
//  CHECK-SAME:     %[[ARG0:.+]]: i1
//       CHECK:   %[[RESULT:.+]] = flow.dispatch.workgroups(%[[ARG0]])
//  CHECK-NEXT:       %[[ARG1:.+]]: i1, %[[ARG2:.+]]: !flow.dispatch.tensor
//       CHECK:     %[[CST:.+]] = arith.constant
//       CHECK:     ^[[BB1:.+]]:
//       CHECK:       %[[EMPTY:.+]] = tensor.empty()
//       CHECK:       flow.dispatch.tensor.store %[[EMPTY]], %[[ARG2]]
//       CHECK:       flow.return
//       CHECK:     ^[[BB2:.+]]:
//       CHECK:       flow.dispatch.tensor.store %[[CST]], %[[ARG2]]
//       CHECK:       flow.return
//       CHECK:   util.return %[[RESULT]]
