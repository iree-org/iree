// RUN: iree-opt --allow-unregistered-dialect --split-input-file --iree-flow-convert-to-flow %s | FileCheck %s

// CHECK:  util.func public @tensor.from_elements__to__flow.tensor.splat(%[[arg0:.*]]: i8)
 util.func public @tensor.from_elements__to__flow.tensor.splat(%arg0: i8) -> (i8) {
  // CHECK: %[[splat_res:.*]] = flow.tensor.splat %[[arg0]] : tensor<1xi8>
  %0 = tensor.from_elements %arg0 : tensor<1xi8>
  // CHECK: flow.tensor.load %[[splat_res]]
  %1 = flow.tensor.load %0 : tensor<1xi8>
  util.return %1 : i8
}

// -----
// CHECK:  util.func public @tensor.from_elements__not_convertible(%[[arg0:.*]]: i8)
 util.func public @tensor.from_elements__not_convertible(%arg0: i8) -> (i8) {
  // CHECK: %[[c0:.*]] = arith.constant 0
  %c0 = arith.constant 0 : index
  // CHECK: %[[res:.*]] = tensor.from_elements %[[arg0]], %[[arg0]] : tensor<2xi8>
  %0 = tensor.from_elements %arg0, %arg0 : tensor<2xi8>
  // CHECK: flow.tensor.load %[[res]][%[[c0]]]
  %1 = flow.tensor.load %0[%c0] : tensor<2xi8>
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
//      CHECK:  util.func public @tensor.from_elements_0D
// CHECK-SAME:     %[[ARG0:.+]]: f32
//      CHECK:   %[[SPLAT:.+]] = flow.tensor.splat %[[ARG0]] : tensor<f32>
//      CHECK:   util.return %[[SPLAT]]
