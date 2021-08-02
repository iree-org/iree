// RUN: iree-opt -allow-unregistered-dialect -split-input-file -iree-flow-convert-to-flow-tensor-ops-pass %s | IreeFileCheck %s

// CHECK: func @tensor.from_elements__to__flow.tensor.splat(%[[arg0:.*]]: i8)
func @tensor.from_elements__to__flow.tensor.splat(%arg0: i8) -> (i8) {
  // CHECK: %[[splat_res:.*]] = flow.tensor.splat %[[arg0]]
  %0 = tensor.from_elements %arg0 : tensor<1xi8>
  // CHECK: flow.tensor.load %[[splat_res]]
  %1 = flow.tensor.load %0 : tensor<1xi8>
  return %1 : i8
}

// -----
// CHECK: func @tensor.from_elements__not_convertible(%[[arg0:.*]]: i8)
func @tensor.from_elements__not_convertible(%arg0: i8) -> (i8) {
  // CHECK: %[[c0:.*]] = constant 0
  %c0 = constant 0 : index
  // CHECK: %[[res:.*]] = tensor.from_elements %[[arg0]], %[[arg0]] : tensor<2xi8>
  %0 = tensor.from_elements %arg0, %arg0 : tensor<2xi8>
  // CHECK: flow.tensor.load %[[res]][%[[c0]]]
  %1 = flow.tensor.load %0[%c0] : tensor<2xi8>
  return %1 : i8
}

// -----
func @tensor.from_elements__within_dispatch_workgroups_not_converted() -> tensor<f32> {
  %x = constant 100 : index
  %0 = flow.dispatch.workgroups[%x]() : () -> (tensor<f32>) = () {
    // CHECK: = tensor.from_elements %[[source:.+]] : tensor<1xi8>
    %1 = "test.source"() : () -> (i8)
    %2 = tensor.from_elements %1 : tensor<1xi8>
    "test.sink"(%2) : (tensor<1xi8>) -> ()
    flow.return
  }
  return %0 : tensor<f32>
}
