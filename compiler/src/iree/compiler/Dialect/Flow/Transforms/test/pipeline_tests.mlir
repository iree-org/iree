// RUN: iree-opt --iree-flow-transformation-pipeline --split-input-file --mlir-print-local-scope %s | FileCheck %s

util.func public @scf_for_with_empty_tensor$dynamic_dim_resolution(
    %arg0: !hal.buffer_view) -> !hal.buffer_view {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %input_dim = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  %input = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<?xi64>{%input_dim}
  %empty = tensor.empty(%input_dim) : tensor<?xi64>
  %loop = scf.for %arg1 = %c0 to %input_dim step %c1 iter_args(%arg2 = %empty) -> (tensor<?xi64>)  : index {
    %extracted = flow.tensor.load %input[%arg1] : tensor<?xi64>{%input_dim}
    %dim_0 = tensor.dim %arg2, %c0 : tensor<?xi64>
    %6 = flow.tensor.store %extracted, %arg2[%arg1] : tensor<?xi64>{%dim_0}
    scf.yield %6 : tensor<?xi64>
  }
  %output_dim = tensor.dim %loop, %c0 : tensor<?xi64>
  %output = hal.tensor.export %loop "output0" : tensor<?xi64>{%output_dim} -> !hal.buffer_view
  util.return %output : !hal.buffer_view
}

// CHECK-LABEL: util.func public @scf_for_with_empty_tensor$dynamic_dim_resolution(
//  CHECK-SAME: %[[IN_BUFFER:.*]]: !hal.buffer_view
//       CHECK: %[[CST0:.*]] = arith.constant 0 : index
//       CHECK: %[[IN_DIM:.*]] = hal.buffer_view.dim<%[[IN_BUFFER]] : !hal.buffer_view>[0] : index
//       CHECK: %[[IMPORT:.*]] = hal.tensor.import %[[IN_BUFFER]]
//  CHECK-SAME: !hal.buffer_view -> tensor<?xi64>{%[[IN_DIM]]}
//       CHECK: %[[EMPTY:.*]] = flow.tensor.empty : tensor<?xi64>{%[[IN_DIM]]}
//       CHECK: %[[LOOP:.*]] = scf.for %[[INDEX:.*]] = %[[CST0]]
//  CHECK-SAME: iter_args(%[[ITER_TENSOR:.*]] = %[[EMPTY]]) -> (tensor<?xi64>)
//       CHECK:   %[[LOAD:.*]] = flow.tensor.load %[[IMPORT]][%[[INDEX]]] : tensor<?xi64>{%[[IN_DIM]]}
//       CHECK:   %[[STORE:.*]] = flow.tensor.store %[[LOAD]], %[[ITER_TENSOR]][%[[INDEX]]] : tensor<?xi64>{%[[IN_DIM]]}
//       CHECK: %[[EXPORT:.*]] = hal.tensor.export %[[LOOP]]
//  CHECK-SAME: tensor<?xi64>{%[[IN_DIM]]} -> !hal.buffer_view
//       CHECK: util.return %[[EXPORT]] : !hal.buffer_view
