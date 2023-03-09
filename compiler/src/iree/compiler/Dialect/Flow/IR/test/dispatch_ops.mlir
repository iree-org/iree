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

// -----

// CHECK-LABEL: @region
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x?xf32>)
func.func @region(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[R:.*]] = flow.dispatch.region -> (tensor<?x?xf32>{%{{.*}}, %{{.*}}}) {
  // CHECK:   flow.return %[[ARG0]] : tensor<?x?xf32>
  // CHECK: }
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %r = flow.dispatch.region -> (tensor<?x?xf32>{%d0, %d1}) {
    flow.return %arg0 : tensor<?x?xf32>
  }
  // CHECK: return %[[R]]
  return %r : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @regionStaticShape
// CHECK-SAME: (%[[ARG0:.+]]: tensor<5x10xf32>)
func.func @regionStaticShape(%arg0: tensor<5x10xf32>) -> tensor<5x10xf32> {
  // CHECK: %[[R:.*]] = flow.dispatch.region -> (tensor<5x10xf32>) {
  // CHECK:   flow.return %[[ARG0]] : tensor<5x10xf32>
  // CHECK: }
  %r = flow.dispatch.region -> (tensor<5x10xf32>) {
    flow.return %arg0 : tensor<5x10xf32>
  }
  // CHECK: return %[[R]]
  return %r : tensor<5x10xf32>
}

// -----

// CHECK-LABEL: func.func @regionDynamicShape
// CHECK-SAME: (%[[ARG0:.+]]: tensor<?x?x16xf32>, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index, %[[DIM2:.+]]: index, %[[DIM3:.+]]: index)
func.func @regionDynamicShape(%arg0: tensor<?x?x16xf32>, %dim0: index, %dim1: index, %dim2: index, %dim3: index) -> tensor<?x?x16xf32> {
  // CHECK: %[[C16:.+]] = arith.constant 16 : index
  %c16 = arith.constant 16 : index
  // CHECK: %[[R:.+]] = flow.dispatch.region[%[[DIM0]], %[[DIM1]], %[[C16]]] -> (tensor<?x?x16xf32>{%[[DIM2]], %[[DIM3]]}) {
  // CHECK:   flow.return %[[ARG0]] : tensor<?x?x16xf32>
  // CHECK: }
  %region = flow.dispatch.region[%dim0, %dim1, %c16] -> (tensor<?x?x16xf32>{%dim2, %dim3}) {
    flow.return %arg0 : tensor<?x?x16xf32>
  }
  // CHECK: return %[[R]]
  return %region: tensor<?x?x16xf32>
}

// -----

// CHECK-LABEL: @dynamicizeDim
func.func @dynamicizeDim() -> index {
  // CHECK: flow.dispatch.dynamicize_dim 5 : index
  %0 = flow.dispatch.dynamicize_dim 5 : index
  return %0 : index
}

// -----

// CHECK-LABEL: @dynamicizeShape4D
// CHECK-SAME: (%[[SRC:.+]]: tensor<6x?x7x16xf32>, %[[DIM1:.+]]: index)
func.func @dynamicizeShape4D(%source: tensor<6x?x7x16xf32>, %dim1 : index) -> tensor<?x?x?x?xf32> {
  // CHECK: %[[C6:.+]] = flow.dispatch.dynamicize_dim 6 : index
  %c6 = flow.dispatch.dynamicize_dim 6 : index
  // CHECK: %[[C7:.+]] = flow.dispatch.dynamicize_dim 7 : index
  %c7 = flow.dispatch.dynamicize_dim 7 : index
  // CHECK: %[[C16:.+]] = flow.dispatch.dynamicize_dim 16 : index
  %c16 = flow.dispatch.dynamicize_dim 16 : index
  // CHECK: flow.dispatch.dynamicize_shape %[[SRC]] : tensor<6x?x7x16xf32> -> tensor<?x?x?x?xf32>{%[[C6]], %[[DIM1]], %[[C7]], %[[C16]]}
  %0 = flow.dispatch.dynamicize_shape %source : tensor<6x?x7x16xf32> -> tensor<?x?x?x?xf32>{%c6, %dim1, %c7, %c16}
  return %0 : tensor<?x?x?x?xf32>
}

// -----

// CHECK-LABEL: @dynamicizeShape0D
func.func @dynamicizeShape0D(%source: tensor<f32>) -> tensor<f32> {
  // CHECK: flow.dispatch.dynamicize_shape %{{.+}} : tensor<f32> -> tensor<f32>
  %0 = flow.dispatch.dynamicize_shape %source : tensor<f32> -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @dynamicizeShapeMismatchedSize(%source: tensor<6x?x7x16xf32>, %dim1 : index) -> tensor<?x?x?xf32> {
  %c6 = flow.dispatch.dynamicize_dim 6 : index
  %c7 = flow.dispatch.dynamicize_dim 7 : index
  %c16 = flow.dispatch.dynamicize_dim 16 : index
  // expected-error @+1 {{output rank does not match input rank}}
  %0 = flow.dispatch.dynamicize_shape %source : tensor<6x?x7x16xf32> -> tensor<?x?x?xf32>{%c6, %dim1, %c7}
  return %0 : tensor<?x?x?xf32>
}

// -----

func.func @dynamicizeShapeMismatchedSize(%source: tensor<6x?x7x16xf32>, %dim1 : index) -> tensor<?x?x?x?xf32> {
  %c6 = flow.dispatch.dynamicize_dim 6 : index
  %c7 = flow.dispatch.dynamicize_dim 7 : index
  // expected-error @+1 {{dynamic shape value count does not match output rank}}
  %0 = flow.dispatch.dynamicize_shape %source : tensor<6x?x7x16xf32> -> tensor<?x?x?x?xf32>{%c6, %dim1, %c7}
  return %0 : tensor<?x?x?x?xf32>
}

// -----

func.func @dynamicizeShapeOutput(%source: tensor<6x?x7x16xf32>, %dim1 : index) -> tensor<?x?x?x16xf32> {
  %c6 = flow.dispatch.dynamicize_dim 6 : index
  %c7 = flow.dispatch.dynamicize_dim 7 : index
  %c16 = flow.dispatch.dynamicize_dim 16 : index
  // expected-error @+1 {{output shape should be fully dynamic}}
  %0 = flow.dispatch.dynamicize_shape %source : tensor<6x?x7x16xf32> -> tensor<?x?x?x16xf32>{%c6, %dim1, %c7, %c16}
  return %0 : tensor<?x?x?x16xf32>
}

// -----

func.func @dynamicizeShapeWrongDimOp(%source: tensor<6xf32>) -> tensor<?xf32> {
  %c6 = arith.constant 6 : index
  // expected-error @+1 {{output dimension SSA value should come from flow.dispatch.dynamicize_dim}}
  %0 = flow.dispatch.dynamicize_shape %source : tensor<6xf32> -> tensor<?xf32>{%c6}
  return %0 : tensor<?xf32>
}

// -----

func.func @dynamicizeShapeMismatchedStaticDim(%source: tensor<6xf32>) -> tensor<?xf32> {
  %c8 = flow.dispatch.dynamicize_dim 8 : index
  // expected-error @+1 {{input dimension size 6 expected the corresponding output dimension to capture the same constant SSA value}}
  %0 = flow.dispatch.dynamicize_shape %source : tensor<6xf32> -> tensor<?xf32>{%c8}
  return %0 : tensor<?xf32>
}
