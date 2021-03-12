// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

flow.executable @ex0 {
  module {
    func @dispatch_fn(%cst : index, %arg0 : tensor<4xf32>) -> tensor<4xf32> {
      return %arg0 : tensor<4xf32>
    }
  }
  flow.dispatch.entry @dispatch_fn
}

// CHECK-LABEL: @dispatch
func @dispatch(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[CST:.+]] = constant
  %cst = constant 4 : index
  // CHECK: %0 = flow.dispatch @ex0::@dispatch_fn[%[[CST]]](%[[CST]], %arg0) : (index, tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @ex0::@dispatch_fn[%cst](%cst, %arg0) : (index, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @inplaceDispatch
func @inplaceDispatch(%arg0 : tensor<4xf32>, %arg1 : tensor<8xf32>) -> (tensor<4xf32>, tensor<8xf32>) {
  // CHECK: %[[CST:.+]] = constant
  %cst = constant 4 : index
  // CHECK: %0:2 = flow.dispatch @ex0::@dispatch_fn[%[[CST]]](%[[CST]], %arg0, %arg1) : (index, tensor<4xf32>, tensor<8xf32>) -> (%arg0, %arg1)
  %0, %1 = flow.dispatch @ex0::@dispatch_fn[%cst](%cst, %arg0, %arg1) : (index, tensor<4xf32>, tensor<8xf32>) -> (%arg0, %arg1)
  return %0, %1 : tensor<4xf32>, tensor<8xf32>
}

// -----

// CHECK-LABEL: @inplaceDynamicDispatch
func @inplaceDynamicDispatch(%arg0 : tensor<4x?xf32>, %arg1 : tensor<8x?xf32>) -> (tensor<4x?xf32>, tensor<8x?xf32>) {
  // CHECK-DAG: %[[CST:.+]] = constant 4
  %cst = constant 4 : index
  // CHECK-DAG: %[[DIM0:.+]] = constant 100
  %dim0 = constant 100 : index
  // CHECK-DAG: %[[DIM1:.+]] = constant 200
  %dim1 = constant 200 : index
  // CHECK: %0:2 = flow.dispatch @ex0::@dispatch_fn[%[[CST]]](%[[CST]], %arg0, %arg1) : (index, tensor<4x?xf32>{%[[DIM0]]}, tensor<8x?xf32>{%[[DIM1]]}) -> (%arg0, %arg1)
  %0, %1 = flow.dispatch @ex0::@dispatch_fn[%cst](%cst, %arg0, %arg1) : (index, tensor<4x?xf32>{%dim0}, tensor<8x?xf32>{%dim1}) -> (%arg0, %arg1)
  return %0, %1 : tensor<4x?xf32>, tensor<8x?xf32>
}
