// RUN: iree-opt --allow-unregistered-dialect --split-input-file --iree-flow-convert-to-flow %s | FileCheck %s

 util.func public @insert_convert_zero_ranked_tensor
    (%arg0 : tensor<i64>) -> tensor<i64> {
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.insert %c0_i64 into %arg0[] : tensor<i64>
  util.return %0 : tensor<i64>
}
// CHECK-LABEL:  util.func public @insert_convert_zero_ranked_tensor
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0_I64:.+]] = arith.constant 0 : i64
//       CHECK:   %[[UPDATE:.+]] = flow.tensor.store %[[C0_I64]], %[[ARG0]] : tensor<i64>

// -----

 util.func public @insert_convert
    (%arg0 : tensor<2x3xi64>) -> tensor<2x3xi64> {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.insert %c0_i64 into %arg0[%c0, %c0] : tensor<2x3xi64>
  util.return %0 : tensor<2x3xi64>
}
// CHECK-LABEL:  util.func public @insert_convert
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C0_I64:.+]] = arith.constant 0 : i64
//       CHECK:   %[[UPDATE:.+]] = flow.tensor.store %[[C0_I64]], %[[ARG0]][%[[C0]], %[[C0]]] : tensor<2x3xi64>

// -----

 util.func public @insert_convert_dynamic_dims
    (%arg0 : tensor<?x3xi64>) -> tensor<?x3xi64> {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %0 = tensor.insert %c0_i64 into %arg0[%c0, %c0] : tensor<?x3xi64>
  util.return %0 : tensor<?x3xi64>
}
// CHECK-LABEL:  util.func public @insert_convert_dynamic_dims
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C0_I64:.+]] = arith.constant 0 : i64
//   CHECK-DAG:   %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//       CHECK:   %[[UPDATE:.+]] = flow.tensor.store %[[C0_I64]], %[[ARG0]][%[[C0]], %[[C0]]] : tensor<?x3xi64>

// -----

 util.func public @insert_within_dispatch_workgroups_not_converted() -> tensor<f32> {
  %x = arith.constant 100 : index
  %0 = flow.dispatch.workgroups[%x]() : () -> (tensor<f32>) = () {
    %c0 = arith.constant 0 : index
    %c100_i64 = arith.constant 100 : i64
    %1 = "test.source1"() : () -> (tensor<2x3xi64>)
    // CHECK: = tensor.insert %[[CST100_I64:.+]] into %[[SOURCE1:.+]][%[[INDEX:.+]], %[[INDEX:.+]]] : tensor<2x3xi64>
    %2 = tensor.insert %c100_i64 into %1[%c0, %c0] : tensor<2x3xi64>
    "test.sink"(%2) : (tensor<2x3xi64>) -> ()
    flow.return
  }
  util.return %0 : tensor<f32>
}
