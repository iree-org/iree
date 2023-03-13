// RUN: iree-opt --pass-pipeline='builtin.module(iree-abi-wrap-entry-points{invocation-model=coarse-fences})' --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @asyncEntry(
//  CHECK-SAME:   %[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view, %[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence
//  CHECK-SAME: -> (
//  CHECK-SAME:   !hal.buffer_view, !hal.buffer_view
//  CHECK-SAME: ) attributes {
//  CHECK-SAME:   iree.abi.stub
//  CHECK-SAME:   iree.reflection = {iree.abi.model = "coarse-fences"}
//  CHECK-SAME: } {
//  CHECK-NEXT:   %[[ARG0_TENSOR:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[ARG0]] "input 0" : !hal.buffer_view -> tensor<4xf32>
//  CHECK-NEXT:   %[[ARG1_TENSOR:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[ARG1]] "input 1" : !hal.buffer_view -> tensor<4xf32>
//  CHECK-NEXT:   %[[RESULT_TENSORS:.+]]:2 = call @_asyncEntry(%[[ARG0_TENSOR]], %[[ARG1_TENSOR]])
//  CHECK-NEXT:   %[[READY_TENSORS:.+]]:2 = hal.tensor.barrier join(%[[RESULT_TENSORS]]#0, %[[RESULT_TENSORS]]#1 : tensor<4xf32>, tensor<4xf32>) => %[[SIGNAL]] : !hal.fence
//  CHECK-NEXT:   %[[RET0_VIEW:.+]] = hal.tensor.export %[[READY_TENSORS]]#0 "output 0" : tensor<4xf32> -> !hal.buffer_view
//  CHECK-NEXT:   %[[RET1_VIEW:.+]] = hal.tensor.export %[[READY_TENSORS]]#1 "output 1" : tensor<4xf32> -> !hal.buffer_view
//  CHECK-NEXT:   return %[[RET0_VIEW]], %[[RET1_VIEW]] : !hal.buffer_view, !hal.buffer_view
//  CHECK-NEXT: }

// CHECK-LABEL: func.func private @_asyncEntry(
func.func @asyncEntry(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
  %1 = arith.addf %0, %arg0 : tensor<4xf32>
  return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @bareFunc
//  CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence)
//       CHECK:   call @_bareFunc()
//  CHECK-NEXT:   hal.fence.signal<%[[SIGNAL]] : !hal.fence>
//  CHECK-NEXT:   return

// CHECK-LABEL: func.func private @_bareFunc(
func.func @bareFunc() {
  return
}

// -----

// CHECK-LABEL: func.func @primitiveArgOnly
//  CHECK-SAME: (%[[ARG0:.+]]: i32, %[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence)
//  CHECK-NEXT:   call @_primitiveArgOnly(%[[ARG0]])
//  CHECK-NEXT:   hal.fence.signal<%[[SIGNAL]] : !hal.fence>
//  CHECK-NEXT:   return

// CHECK-LABEL: func.func private @_primitiveArgOnly(
func.func @primitiveArgOnly(%arg0: i32) {
  %0 = arith.addi %arg0, %arg0 : i32
  util.optimization_barrier %0 : i32
  return
}

// -----

// CHECK-LABEL: func.func @tensorArgOnly
//  CHECK-SAME: (%[[ARG0:.+]]: !hal.buffer_view, %[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence)
//       CHECK:   %[[ARG0_TENSOR:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[ARG0]] "input 0" : !hal.buffer_view -> tensor<4xf32>
//  CHECK-NEXT:   call @_tensorArgOnly(%[[ARG0_TENSOR]])
//  CHECK-NEXT:   hal.fence.signal<%[[SIGNAL]] : !hal.fence>
//  CHECK-NEXT:   return

// CHECK-LABEL: func.func private @_tensorArgOnly(
func.func @tensorArgOnly(%arg0: tensor<4xf32>) {
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.optimization_barrier %0 : tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @primitiveResultOnly
//  CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence) -> i32
//  CHECK-NEXT:   %[[RESULT:.+]] = call @_primitiveResultOnly()
//  CHECK-NEXT:   hal.fence.signal<%[[SIGNAL]] : !hal.fence>
//  CHECK-NEXT:   return %[[RESULT]]

// CHECK-LABEL: func.func private @_primitiveResultOnly(
func.func @primitiveResultOnly() -> i32 {
  %0 = arith.constant 8 : i32
  %1 = util.optimization_barrier %0 : i32
  return %1 : i32
}

// -----

// CHECK-LABEL: func.func @tensorResultOnly
//  CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence) -> !hal.buffer_view
//  CHECK-NEXT:   %[[RESULT_TENSOR:.+]] = call @_tensorResultOnly()
//  CHECK-NEXT:   %[[READY_TENSOR:.+]] = hal.tensor.barrier join(%[[RESULT_TENSOR]] : tensor<4xf32>) => %[[SIGNAL]] : !hal.fence
//  CHECK-NEXT:   %[[RESULT_VIEW:.+]] = hal.tensor.export %[[READY_TENSOR]]
//  CHECK-NEXT:   return %[[RESULT_VIEW]]

// CHECK-LABEL: func.func private @_tensorResultOnly(
func.func @tensorResultOnly() -> tensor<4xf32> {
  %0 = arith.constant dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf32>
  %1 = util.optimization_barrier %0 : tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

// Tests that imported functions with the coarse-fences execution model
// specified get wrapped with fences. Note that unlike exports controlled by
// compiler flags imports only get the fences when explicitly specified so as
// that is part of their ABI. Users can always manually specify the fences too
// though that's much more verbose.

// CHECK-LABEL: func.func private @import(!hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> (!hal.buffer_view, !hal.buffer_view)
func.func private @import(tensor<?x2xi32>, tensor<?x3xi32>) -> (tensor<2x?xi32>, tensor<3x?xi32>) attributes {
  iree.abi.model = "coarse-fences",
  nosideeffects
}

// CHECK: func.func private @_import(%[[ARG0_TENSOR:.+]]: tensor<?x2xi32>, %[[ARG1_TENSOR:.+]]: tensor<?x3xi32>) -> (tensor<2x?xi32>, tensor<3x?xi32>) {

// Prepare fences and put a barrier on input arguments:
// CHECK:   %[[DEVICE:.+]] = hal.ex.shared_device
// CHECK:   %[[WAIT_FENCE:.+]] = hal.fence.create device(%[[DEVICE]]
// CHECK:   %[[ARG_BARRIER:.+]]:2 = hal.tensor.barrier join(%[[ARG0_TENSOR]], %[[ARG1_TENSOR]] : tensor<?x2xi32>, tensor<?x3xi32>) => %[[WAIT_FENCE]] : !hal.fence
// CHECK:   %[[SIGNAL_FENCE:.+]] = hal.fence.create device(%[[DEVICE]]

// Export input arguments to buffer views:
// CHECK:   %[[ARG0_DIM:.+]] = tensor.dim %[[ARG_BARRIER]]#0, %c0
// CHECK:   %[[ARG0_VIEW:.+]] = hal.tensor.export %[[ARG_BARRIER]]#0 : tensor<?x2xi32>{%[[ARG0_DIM]]} -> !hal.buffer_view
// CHECK:   %[[ARG1_DIM:.+]] = tensor.dim %[[ARG_BARRIER]]#1, %c0
// CHECK:   %[[ARG1_VIEW:.+]] = hal.tensor.export %[[ARG_BARRIER]]#1 : tensor<?x3xi32>{%[[ARG1_DIM]]} -> !hal.buffer_view

// Call the import:
// CHECK:   %[[RET_VIEWS:.+]]:2 = call @import(%[[ARG0_VIEW]], %[[ARG1_VIEW]], %[[WAIT_FENCE]], %[[SIGNAL_FENCE]]) : (!hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> (!hal.buffer_view, !hal.buffer_view)

// Import output results from buffer views:
// CHECK:   %[[RET0_DIM:.+]] = hal.buffer_view.dim<%[[RET_VIEWS]]#0 : !hal.buffer_view>[1]
// CHECK:   %[[RET0_TENSOR:.+]] = hal.tensor.import wait(%[[SIGNAL_FENCE]]) => %[[RET_VIEWS]]#0 : !hal.buffer_view -> tensor<2x?xi32>{%[[RET0_DIM]]}
// CHECK:   %[[RET1_DIM:.+]] = hal.buffer_view.dim<%[[RET_VIEWS]]#1 : !hal.buffer_view>[1]
// CHECK:   %[[RET1_TENSOR:.+]] = hal.tensor.import wait(%[[SIGNAL_FENCE]]) => %[[RET_VIEWS]]#1 : !hal.buffer_view -> tensor<3x?xi32>{%[[RET1_DIM]]}

// CHECK:   return %[[RET0_TENSOR]], %[[RET1_TENSOR]] : tensor<2x?xi32>, tensor<3x?xi32>
// CHECK: }

// CHECK: func.func private @caller(%[[ARG0_CALLER:.+]]: tensor<?x2xi32>, %[[ARG1_CALLER:.+]]: tensor<?x3xi32>)
func.func private @caller(%arg0: tensor<?x2xi32>, %arg1: tensor<?x3xi32>) -> (tensor<2x?xi32>, tensor<3x?xi32>) {
  // CHECK: %[[RESULTS:.+]]:2 = call @_import(%[[ARG0_CALLER]], %[[ARG1_CALLER]]) : (tensor<?x2xi32>, tensor<?x3xi32>) -> (tensor<2x?xi32>, tensor<3x?xi32>)
  %results:2 = call @import(%arg0, %arg1) : (tensor<?x2xi32>, tensor<?x3xi32>) -> (tensor<2x?xi32>, tensor<3x?xi32>)
  // CHECK-NEXT: return %[[RESULTS]]#0, %[[RESULTS]]#1
  return %results#0, %results#1 : tensor<2x?xi32>, tensor<3x?xi32>
}

// -----

// Tests a side-effect-free import that doesn't take/return reference types.

// CHECK-LABEL: func.func private @importI32(i32, !hal.fence, !hal.fence) -> i32
func.func private @importI32(i32) -> i32 attributes {
  iree.abi.model = "coarse-fences",
  nosideeffects
}

// No fences required as the call has no side-effects and no async resources.
// CHECK: func.func private @_importI32(%[[ARG0:.+]]: i32) -> i32 {
// CHECK:   %[[WAIT_FENCE:.+]] = util.null : !hal.fence
// CHECK:   %[[SIGNAL_FENCE:.+]] = util.null : !hal.fence
// CHECK:   %[[RET0:.+]] = call @importI32(%[[ARG0]], %[[WAIT_FENCE]], %[[SIGNAL_FENCE]]) : (i32, !hal.fence, !hal.fence) -> i32
// CHECK:   return %[[RET0]] : i32
// CHECK: }

// CHECK: func.func private @callerI32(%[[ARG0_CALLER:.+]]: i32)
func.func private @callerI32(%arg0: i32) -> i32 {
  // CHECK: %[[RESULT:.+]] = call @_importI32(%[[ARG0_CALLER]]) : (i32) -> i32
  %result = call @importI32(%arg0) : (i32) -> i32
  // CHECK-NEXT: return %[[RESULT]]
  return %result : i32
}

// -----

// Tests a side-effecting import that requires a host-side wait.

// CHECK-LABEL: func.func private @importI32Effects(!hal.buffer_view, !hal.fence, !hal.fence) -> i32
func.func private @importI32Effects(tensor<4xf32>) -> i32 attributes {
  iree.abi.model = "coarse-fences"
}

// CHECK: func.func private @_importI32Effects(%[[ARG0_TENSOR:.+]]: tensor<4xf32>) -> i32 {

// Wait for the inputs to be ready and create the signal fence to wait on.
// CHECK:   %[[DEVICE:.+]] = hal.ex.shared_device
// CHECK:   %[[WAIT_FENCE:.+]] = hal.fence.create device(%[[DEVICE]]
// CHECK:   %[[ARG0_BARRIER:.+]] = hal.tensor.barrier join(%[[ARG0_TENSOR]] : tensor<4xf32>) => %[[WAIT_FENCE]] : !hal.fence
// CHECK:   %[[SIGNAL_FENCE:.+]] = hal.fence.create device(%[[DEVICE]]

// Marshal inputs:
// CHECK:   %[[ARG0_VIEW:.+]] = hal.tensor.export %[[ARG0_BARRIER]] : tensor<4xf32> -> !hal.buffer_view

// Make the import call:
// CHECK:   %[[RET0:.+]] = call @importI32Effects(%[[ARG0_VIEW]], %[[WAIT_FENCE]], %[[SIGNAL_FENCE]]) : (!hal.buffer_view, !hal.fence, !hal.fence) -> i32

// Perform host-side wait.
// CHECK:   hal.fence.await until([%[[SIGNAL_FENCE]]])

// CHECK:   return %[[RET0]] : i32
// CHECK: }

// CHECK: func.func private @callerI32Effects(%[[ARG0_CALLER:.+]]: tensor<4xf32>)
func.func private @callerI32Effects(%arg0: tensor<4xf32>) -> i32 {
  // CHECK: %[[RESULT:.+]] = call @_importI32Effects(%[[ARG0_CALLER]]) : (tensor<4xf32>) -> i32
  %result = call @importI32Effects(%arg0) : (tensor<4xf32>) -> i32
  // CHECK-NEXT: return %[[RESULT]]
  return %result : i32
}
