// RUN: iree-opt --split-input-file %s --verify-diagnostics | FileCheck %s

flow.executable @ex0 {
  builtin.module {
    func.func @dispatch_fn(%cst : index, %arg0 : tensor<4xf32>) -> tensor<4xf32> {
      return %arg0 : tensor<4xf32>
    }
  }
  flow.executable.export @dispatch_fn
}

// CHECK-LABEL: @dispatch
func.func @dispatch(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[CST:.+]] = arith.constant
  %cst = arith.constant 4 : index
  // CHECK: %0 = flow.dispatch @ex0::@dispatch_fn[%[[CST]]](%[[CST]], %arg0) : (index, tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @ex0::@dispatch_fn[%cst](%cst, %arg0) : (index, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

flow.executable private @ex0 {
  flow.executable.export public @dispatch workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
    flow.return %arg0, %arg1, %arg0 : index, index, index
  }
  builtin.module {
    func.func @dispatch() {
      return
    }
  }
}

// CHECK-LABEL: @asyncDispatchWithWorkgroupCount
func.func @asyncDispatchWithWorkgroupCount(%arg0: tensor<4xf32>, %arg1: index) -> tensor<4xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // CHECK: = flow.dispatch @ex0::@dispatch[%c1, %c2](%arg0, %arg1) : (tensor<4xf32>, index) -> tensor<4xf32>
  %0 = flow.dispatch @ex0::@dispatch[%c1, %c2](%arg0, %arg1) : (tensor<4xf32>, index) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

flow.executable private @ex0 {
  flow.executable.export public @dispatch workgroups(%arg0: index) -> (index, index, index) {
    flow.return %arg0, %arg0, %arg0 : index, index, index
  }
  builtin.module {
    func.func @dispatch() {
      return
    }
  }
}

func.func @asyncDispatchWithInvalidWorkload(%arg0: tensor<4xf32>, %arg1: index) -> tensor<4xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  // expected-error @+1 {{op workload mismatch; entry point expects 1 arguments but dispatch provides 2}}
  %0 = flow.dispatch @ex0::@dispatch[%c1, %c2](%arg0, %arg1) : (tensor<4xf32>, index) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @dispatchNoWorkload
func.func @dispatchNoWorkload(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[CST:.+]] = arith.constant
  %cst = arith.constant 4 : index
  // CHECK: %0 = flow.dispatch @ex0::@dispatch_fn(%[[CST]], %arg0) : (index, tensor<4xf32>) -> tensor<4xf32>
  %0 = flow.dispatch @ex0::@dispatch_fn(%cst, %arg0) : (index, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @inplaceDispatch
func.func @inplaceDispatch(%arg0 : tensor<4xf32>, %arg1 : tensor<8xf32>) -> (tensor<4xf32>, tensor<8xf32>) {
  // CHECK: %[[CST:.+]] = arith.constant
  %cst = arith.constant 4 : index
  // CHECK: %0:2 = flow.dispatch @ex0::@dispatch_fn[%[[CST]]](%[[CST]], %arg0, %arg1) : (index, tensor<4xf32>, tensor<8xf32>) -> (%arg0, %arg1)
  %0, %1 = flow.dispatch @ex0::@dispatch_fn[%cst](%cst, %arg0, %arg1) : (index, tensor<4xf32>, tensor<8xf32>) -> (%arg0, %arg1)
  return %0, %1 : tensor<4xf32>, tensor<8xf32>
}

// -----

// CHECK-LABEL: @inplaceDynamicDispatch
func.func @inplaceDynamicDispatch(%arg0 : tensor<4x?xf32>, %arg1 : tensor<8x?xf32>) -> (tensor<4x?xf32>, tensor<8x?xf32>) {
  // CHECK-DAG: %[[CST:.+]] = arith.constant 4
  %cst = arith.constant 4 : index
  // CHECK-DAG: %[[DIM0:.+]] = arith.constant 100
  %dim0 = arith.constant 100 : index
  // CHECK-DAG: %[[DIM1:.+]] = arith.constant 200
  %dim1 = arith.constant 200 : index
  // CHECK: %0:2 = flow.dispatch @ex0::@dispatch_fn[%[[CST]]](%[[CST]], %arg0, %arg1) : (index, tensor<4x?xf32>{%[[DIM0]]}, tensor<8x?xf32>{%[[DIM1]]}) -> (%arg0{%[[DIM1]]}, %arg1{%[[DIM0]]})
  %0, %1 = flow.dispatch @ex0::@dispatch_fn[%cst](%cst, %arg0, %arg1) : (index, tensor<4x?xf32>{%dim0}, tensor<8x?xf32>{%dim1}) -> (%arg0{%dim1}, %arg1{%dim0})
  return %0, %1 : tensor<4x?xf32>, tensor<8x?xf32>
}

// -----

// CHECK-LABEL: @inplaceTypeChange
// CHECK-SAME: (%[[ARG0:.+]]: tensor<4x?xf32>)
func.func @inplaceTypeChange(%arg0: tensor<4x?xf32>) -> tensor<?x4xf32> {
  // CHECK-DAG: %[[CST:.+]] = arith.constant 4
  %cst = arith.constant 4 : index
  // CHECK-DAG: %[[DIM0:.+]] = arith.constant 100
  %dim0 = arith.constant 100 : index
  // CHECK: %0 = flow.dispatch @ex0::@dispatch_fn[%[[CST]]](%[[ARG0]]) : (tensor<4x?xf32>{%[[DIM0]]}) -> %arg0 as tensor<?x4xf32>{%[[DIM0]]}
  %0 = flow.dispatch @ex0::@dispatch_fn[%cst](%arg0) : (tensor<4x?xf32>{%dim0}) -> %arg0 as tensor<?x4xf32>{%dim0}
  return %0 : tensor<?x4xf32>
}
