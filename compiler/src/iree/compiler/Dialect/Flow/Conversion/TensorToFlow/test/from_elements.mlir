// RUN: iree-opt --allow-unregistered-dialect --split-input-file --iree-flow-convert-to-flow %s | FileCheck %s

// CHECK: util.func public @tensor.from_elements__to__flow.tensor.splat(%[[arg0:.*]]: i8)
util.func public @tensor.from_elements__to__flow.tensor.splat(%arg0: i8) -> (i8) {
  // CHECK: %[[splat_res:.*]] = flow.tensor.splat %[[arg0]] : tensor<1xi8>
  %0 = tensor.from_elements %arg0 : tensor<1xi8>
  // CHECK: flow.tensor.load %[[splat_res]]
  %1 = flow.tensor.load %0 : tensor<1xi8>
  util.return %1 : i8
}

// -----
util.func public @tensor.from_elements__within_dispatch_workgroups_not_converted() -> tensor<f32> {
  %x = arith.constant 100 : index
  %0 = flow.dispatch.workgroups[%x]() : () -> (tensor<f32>) = () {
    // CHECK: = tensor.from_elements %[[source:.+]] : tensor<1xi8>
    %1 = "test.source"() : () -> (i8)
    %2 = tensor.from_elements %1 : tensor<1xi8>
    "test.sink"(%2) : (tensor<1xi8>) -> ()
    flow.return
  }
  util.return %0 : tensor<f32>
}

// -----

util.func public @tensor.from_elements_0D(%arg0 : f32) -> tensor<f32> {
  %0 = tensor.from_elements %arg0 : tensor<f32>
  util.return %0 : tensor<f32>
}
//      CHECK: util.func public @tensor.from_elements_0D
// CHECK-SAME:     %[[ARG0:.+]]: f32
//      CHECK:   %[[SPLAT:.+]] = flow.tensor.splat %[[ARG0]] : tensor<f32>
//      CHECK:   util.return %[[SPLAT]]

// -----

util.func @tensor.from_elements_2D(%arg0 : f32, %arg1 : f32, %arg2 : f32, %arg3 : f32, %arg4 : f32, %arg5 : f32) -> tensor<2x3xf32> {
  %0 = tensor.from_elements %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : tensor<2x3xf32>
  util.return %0 : tensor<2x3xf32>
}

//      CHECK: util.func public @tensor.from_elements_2D
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[EMPTY:.+]] = tensor.empty() : tensor<2x3xf32>
// CHECK-DAG: %[[STORE0:.+]] = flow.tensor.store %arg0, %[[EMPTY]][%[[C0]], %[[C0]]] : tensor<2x3xf32>
// CHECK-DAG: %[[STORE1:.+]] = flow.tensor.store %arg1, %[[STORE0]][%[[C0]], %[[C1]]] : tensor<2x3xf32>
// CHECK-DAG: %[[STORE2:.+]] = flow.tensor.store %arg2, %[[STORE1]][%[[C0]], %[[C2]]] : tensor<2x3xf32>
// CHECK-DAG: %[[STORE3:.+]] = flow.tensor.store %arg3, %[[STORE2]][%[[C1]], %[[C0]]] : tensor<2x3xf32>
// CHECK-DAG: %[[STORE4:.+]] = flow.tensor.store %arg4, %[[STORE3]][%[[C1]], %[[C1]]] : tensor<2x3xf32>
// CHECK:     %[[STORE5:.+]] = flow.tensor.store %arg5, %[[STORE4]][%[[C1]], %[[C2]]] : tensor<2x3xf32>
