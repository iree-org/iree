// RUN: iree-opt -resolve-ranked-shaped-type-result-dims -split-input-file %s | FileCheck %s

// CHECK-LABEL: @tensor_load_op
//  CHECK-SAME: (%[[DIM0:.+]]: index, %[[DIM1:.+]]: index
util.func public @tensor_load_op(%dim0: index, %dim1: index, %binding: !flow.dispatch.tensor<readonly:tensor<?x1x1x?xf32>>) -> (index, index) {
  %tensor = flow.dispatch.tensor.load %binding, offsets = [0, 0, 0, 0], sizes = [%dim0, 1, 1, %dim1], strides = [1, 1, 1, 1]
      : !flow.dispatch.tensor<readonly:tensor<?x1x1x?xf32>>{%dim0, %dim1} -> tensor<?x?xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %queried_dim0 = tensor.dim %tensor, %c0 : tensor<?x?xf32>
  %queried_dim1 = tensor.dim %tensor, %c1 : tensor<?x?xf32>
  // CHECK: util.return %[[DIM0]], %[[DIM1]]
  util.return %queried_dim0, %queried_dim1 : index, index
}
