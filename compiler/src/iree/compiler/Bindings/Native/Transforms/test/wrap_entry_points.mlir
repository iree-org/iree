// RUN: iree-opt --pass-pipeline='builtin.module(iree-abi-wrap-entry-points{invocation-model=sync})' --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @dynamicEntry(
//  CHECK-SAME:   %[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view
//  CHECK-SAME: -> (
//  CHECK-SAME:   !hal.buffer_view, !hal.buffer_view
//  CHECK-SAME: ) attributes {
//  CHECK-SAME:   iree.abi.stub
//  CHECK-SAME: } {
//  CHECK-NEXT:   %[[ARG0_DIM0:.+]] = hal.buffer_view.dim<%[[ARG0]] : !hal.buffer_view>[0] : index
//  CHECK-NEXT:   %[[ARG0_TENSOR:.+]] = hal.tensor.import %[[ARG0]] "input 0" : !hal.buffer_view -> tensor<?x8x8x3xf32>{%[[ARG0_DIM0]]}
//  CHECK-NEXT:   %[[ARG1_DIM0:.+]] = hal.buffer_view.dim<%[[ARG1]] : !hal.buffer_view>[0] : index
//  CHECK-NEXT:   %[[ARG1_TENSOR:.+]] = hal.tensor.import %[[ARG1]] "input 1" : !hal.buffer_view -> tensor<?x8x8x3xf32>{%[[ARG1_DIM0]]}
//  CHECK-NEXT:   %[[RET_TENSORS:.+]]:2 = call @_dynamicEntry(%[[ARG0_TENSOR]], %[[ARG1_TENSOR]])
//       CHECK:   %[[RET0_DIM0:.+]] = tensor.dim %[[RET_TENSORS]]#0, %c0{{.*}} : tensor<?x8x8x3xf32>
//  CHECK-NEXT:   %[[RET0_VIEW:.+]] = hal.tensor.export %[[RET_TENSORS]]#0 "output 0" : tensor<?x8x8x3xf32>{%[[RET0_DIM0]]} -> !hal.buffer_view
//       CHECK:   %[[RET1_DIM0:.+]] = tensor.dim %[[RET_TENSORS]]#1, %c0{{.*}} : tensor<?x8x8x3xf32>
//  CHECK-NEXT:   %[[RET1_VIEW:.+]] = hal.tensor.export %[[RET_TENSORS]]#1 "output 1" : tensor<?x8x8x3xf32>{%[[RET1_DIM0]]} -> !hal.buffer_view
//  CHECK-NEXT:   return %[[RET0_VIEW]], %[[RET1_VIEW]] : !hal.buffer_view, !hal.buffer_view
//  CHECK-NEXT: }

// CHECK-LABEL: func.func private @_dynamicEntry(
func.func @dynamicEntry(%arg0: tensor<?x8x8x3xf32>, %arg1: tensor<?x8x8x3xf32>) ->
    (tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>) {
  %0 = arith.addf %arg0, %arg1 : tensor<?x8x8x3xf32>
  %1 = arith.addf %0, %arg0 : tensor<?x8x8x3xf32>
  return %0, %1 : tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>
}

// -----

// CHECK-LABEL: func.func @outputStorage(
//  CHECK-SAME:   %[[ARG0:.+]]: !hal.buffer_view,
//  CHECK-SAME:   %[[RET1_STORAGE:.+]]: !hal.buffer
//  CHECK-SAME: -> (
//  CHECK-SAME:   !hal.buffer_view, !hal.buffer_view
//  CHECK-SAME: ) attributes {
//  CHECK-SAME:   iree.abi.stub
//  CHECK-SAME: } {
//  CHECK-NEXT:   %[[ARG0_DIM0:.+]] = hal.buffer_view.dim<%[[ARG0]] : !hal.buffer_view>[0] : index
//  CHECK-NEXT:   %[[ARG0_TENSOR:.+]] = hal.tensor.import %[[ARG0]] "input 0" : !hal.buffer_view -> tensor<?x8x8x3xf32>{%[[ARG0_DIM0]]}
//  CHECK-NEXT:   %[[RET_TENSORS:.+]]:2 = call @_outputStorage(%[[ARG0_TENSOR]], %[[RET1_STORAGE]])
//       CHECK:   %[[RET0_DIM0:.+]] = tensor.dim %[[RET_TENSORS]]#0, %c0{{.*}} : tensor<?x8x8x3xf32>
//  CHECK-NEXT:   %[[RET0_VIEW:.+]] = hal.tensor.export %[[RET_TENSORS]]#0 "output 0" : tensor<?x8x8x3xf32>{%[[RET0_DIM0]]} -> !hal.buffer_view
//       CHECK:   %[[RET1_DIM0:.+]] = tensor.dim %[[RET_TENSORS]]#1, %c0{{.*}} : tensor<?x8x8x3xf32>
//  CHECK-NEXT:   %[[RET1_VIEW:.+]] = hal.tensor.export %[[RET_TENSORS]]#1 "output 1" into %[[RET1_STORAGE]] : tensor<?x8x8x3xf32>{%[[RET1_DIM0]]} -> !hal.buffer_view
//  CHECK-NEXT:   return %[[RET0_VIEW]], %[[RET1_VIEW]] : !hal.buffer_view, !hal.buffer_view
//  CHECK-NEXT: }

// CHECK-LABEL: func.func private @_outputStorage(
func.func @outputStorage(%arg0: tensor<?x8x8x3xf32>, %ret1: !hal.buffer {iree.abi.output = 1 : index}) ->
    (tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>) {
  %0 = arith.addf %arg0, %arg0 : tensor<?x8x8x3xf32>
  %1 = arith.addf %0, %arg0 : tensor<?x8x8x3xf32>
  return %0, %1 : tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>
}

// -----

// CHECK-LABEL: func.func @wrappedAlready
//  CHECK-SAME: (%arg0: !hal.buffer_view) -> !hal.buffer_view
//  CHECK-SAME: attributes {iree.abi.stub}
func.func @wrappedAlready(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  return %arg0 : !hal.buffer_view
}
// CHECK-NOT: func.func @_wrappedAlready

// -----

// Tests that a function calling an exported function is redirected to the
// original unwrapped call.

// CHECK-LABEL: func.func @exportA(%arg0: !hal.buffer_view) -> !hal.buffer_view
// CHECK:   call @_exportA
// CHECK: func.func private @_exportA(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK:   return %arg0
func.func @exportA(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  return %arg0 : tensor<?x?xi32>
}

// CHECK: func.func @exportB(%arg0: !hal.buffer_view) -> !hal.buffer_view
// CHECK:   call @_exportB
// CHECK: func.func private @_exportB(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32>
// CHECK:   call @_exportA
func.func @exportB(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %0 = call @exportA(%arg0) : (tensor<?x?xi32>) -> tensor<?x?xi32>
  return %0 : tensor<?x?xi32>
}

// -----

// Tests that imported functions get converted to canonical ABI types and
// wrapper functions are built to preserve internal behavior.

// CHECK-LABEL: func.func private @import(!hal.buffer_view) -> !hal.buffer_view
func.func private @import(tensor<?x2xi32>) -> tensor<2x?xi32>

// CHECK: func.func private @_import(%[[ARG_TENSOR:.+]]: tensor<?x2xi32>) -> tensor<2x?xi32> {
// CHECK:   %[[ARG_DIM:.+]] = tensor.dim %[[ARG_TENSOR]], %c0
// CHECK:   %[[ARG_VIEW:.+]] = hal.tensor.export %[[ARG_TENSOR]] : tensor<?x2xi32>{%[[ARG_DIM]]} -> !hal.buffer_view
// CHECK:   %[[RET_VIEW:.+]] = call @import(%[[ARG_VIEW]]) : (!hal.buffer_view) -> !hal.buffer_view
// CHECK:   %[[RET_DIM:.+]] = hal.buffer_view.dim<%[[RET_VIEW]] : !hal.buffer_view>[1]
// CHECK:   %[[RET_TENSOR:.+]] = hal.tensor.import %[[RET_VIEW]] : !hal.buffer_view -> tensor<2x?xi32>{%[[RET_DIM]]}
// CHECK:   return %[[RET_TENSOR]]
// CHECK: }

// CHECK: func.func private @caller(%arg0: tensor
func.func private @caller(%arg0: tensor<?x2xi32>) -> tensor<2x?xi32> {
  // CHECK: call @_import(%arg0) : (tensor<?x2xi32>) -> tensor<2x?xi32>
  %0 = call @import(%arg0) : (tensor<?x2xi32>) -> tensor<2x?xi32>
  return %0 : tensor<2x?xi32>
}
