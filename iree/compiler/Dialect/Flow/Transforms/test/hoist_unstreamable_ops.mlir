// RUN: iree-opt -split-input-file -iree-flow-hoist-unstreamable-ops %s | IreeFileCheck %s

// CHECK-LABEL: @constants(
func @constants() {
  // CHECK-DAG: %[[W:.+]] = constant 1 : index
  // CHECK-DAG: constant 2 : index
  // CHECK-DAG: constant 3 : index
  // CHECK-DAG: constant 4 : index
  // CHECK-DAG: constant 5 : index
  // CHECK-DAG: constant 6 : index
  // CHECK: flow.dispatch @dispatch0::@dispatch0[%[[W]]]() : () -> tensor<f32>
  // CHECK: flow.dispatch @dispatch1::@dispatch1[%[[W]]]() : () -> tensor<f32>
  // CHECK: flow.dispatch @dispatch2::@dispatch2[%[[W]]]() : () -> tensor<f32>
  // CHECK: flow.dispatch @dispatch3::@dispatch3[%[[W]]]() : () -> tensor<f32>
  // CHECK: flow.dispatch @dispatch4::@dispatch4[%[[W]]]() : () -> tensor<f32>
  // CHECK: flow.dispatch @dispatch5::@dispatch5[%[[W]]]() : () -> tensor<f32>
  %w = constant 1 : index
  %d0 = flow.dispatch @dispatch0::@dispatch0[%w]() : () -> tensor<f32>
  %c2 = constant 2 : index
  %d1 = flow.dispatch @dispatch1::@dispatch1[%w]() : () -> tensor<f32>
  %c3 = constant 3 : index
  %d2 = flow.dispatch @dispatch2::@dispatch2[%w]() : () -> tensor<f32>
  %c4 = constant 4 : index
  %d3 = flow.dispatch @dispatch3::@dispatch3[%w]() : () -> tensor<f32>
  %c5 = constant 5 : index
  %d4 = flow.dispatch @dispatch4::@dispatch4[%w]() : () -> tensor<f32>
  %c6 = constant 6 : index
  %d5 = flow.dispatch @dispatch5::@dispatch5[%w]() : () -> tensor<f32>
  return
}

// -----

// CHECK-LABEL: @dynamic_tensor(
// CHECK-SAME: %[[INPUT:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[SHAPE:[a-zA-Z0-9$._-]+]]
func @dynamic_tensor(%input: tensor<?x?xf32>, %shape: !shapex.ranked_shape<[?,?]>) -> (tensor<?x?xf32>, !shapex.ranked_shape<[?,?]>) {
  // CHECK-DAG: %[[W:.+]] = constant 1
  // CHECK-DAG: %[[DIM0:.+]] shapex.ranked_dim %[[SHAPE]][0]
  // CHECK-DAG: %[[DIM1:.+]] shapex.ranked_dim %[[SHAPE]][1]
  // CHECK:     %[[D:.+]] = flow.dispatch
  %w = constant 1 : index
  %dim0 = shapex.ranked_dim %shape[0] : !shapex.ranked_shape<[?,?]> -> index
  %dim1 = shapex.ranked_dim %shape[1] : !shapex.ranked_shape<[?,?]> -> index
  %d = flow.dispatch @dispatch::@dispatch[%w](%input, %dim0, %dim1) : (tensor<?x?xf32>{%dim0, %dim1}, index, index) -> tensor<?x?xf32>{%dim0, %dim1}
  return %d, %shape : tensor<?x?xf32>, !shapex.ranked_shape<[?,?]>
}

// -----

// CHECK-LABEL: @dependencies(
func @dependencies() {
  // CHECK-DAG: constant 1
  // CHECK-DAG: constant 2
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  // CHECK: addi
  %add0 = addi %c1, %c2 : index
  return
}

// -----

// CHECK-LABEL: @dependencies_with_dispatch(
func @dependencies_with_dispatch() {
  // CHECK-DAG: %[[W:.+]] = constant 1
  // CHECK-DAG: constant 2
  // CHECK-DAG: constant dense<3>
  %w = constant 1 : index
  %c2 = constant 2 : index
  %ct3 = constant dense<3> : tensor<i32>
  // CHECK: flow.dispatch
  %d0 = flow.dispatch @dispatch0::@dispatch0[%w]() : () -> tensor<i32>
  // CHECK: addi
  %add0 = addi %d0, %ct3 : tensor<i32>
  return
}
