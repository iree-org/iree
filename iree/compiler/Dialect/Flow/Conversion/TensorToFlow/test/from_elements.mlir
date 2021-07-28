// RUN: iree-opt -split-input-file -iree-tensor-to-flow %s | IreeFileCheck %s

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
