// RUN: iree-opt --pass-pipeline="builtin.module(util.func(iree-global-opt-generalize-linalg-named-ops))" --split-input-file %s | FileCheck %s

util.func public @generalize_op(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %empty = tensor.empty(%d0, %d1): tensor<?x?xf32>
  %add = linalg.add ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  util.return %add : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @generalize_op
//       CHECK:   %[[GENERIC:.+]] = linalg.generic
//       CHECK:   util.return %[[GENERIC]]

// -----

util.func public @no_generalize_op_within_dispatch(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %dispatch = flow.dispatch.region[] -> (tensor<?x?xf32>{%d0, %d1}) {
    %empty = tensor.empty(%d0, %d1): tensor<?x?xf32>
    %add = linalg.add ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    flow.return %add : tensor<?x?xf32>
  }
  util.return %dispatch : tensor<?x?xf32>
}
// CHECK-LABEL: util.func public @no_generalize_op_within_dispatch
//       CHECK:   %[[DISPATCH:.+]] = flow.dispatch.region
//       CHECK:     %[[ADD:.+]] = linalg.add
//       CHECK:     flow.return %[[ADD]]
//       CHECK:   util.return %[[DISPATCH]]

