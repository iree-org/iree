// RUN: iree-opt --pass-pipeline='builtin.module(iree-abi-wrap-entry-points{invocation-model=coarse-fences})' --split-input-file %s | FileCheck %s

// CHECK-LABEL: util.func public @asyncEntry(
//  CHECK-SAME:   %[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view, %[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence
//  CHECK-SAME: -> (
//  CHECK-SAME:   !hal.buffer_view, !hal.buffer_view
//  CHECK-SAME: ) attributes {
//  CHECK-SAME:   iree.abi.stub
//  CHECK-SAME:   iree.reflection =
//  CHECK-SAME:       iree.abi.model = "coarse-fences"
//  CHECK-SAME: } {
//  CHECK-NEXT:   %[[ARG0_TENSOR:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[ARG0]] "input0" : !hal.buffer_view -> tensor<4xf32>
//  CHECK-NEXT:   %[[ARG1_TENSOR:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[ARG1]] "input1" : !hal.buffer_view -> tensor<4xf32>
//  CHECK-NEXT:   %[[RESULT_TENSORS:.+]]:2 = util.call @_asyncEntry(%[[ARG0_TENSOR]], %[[ARG1_TENSOR]])
//  CHECK-NEXT:   %[[READY_TENSORS:.+]]:2 = hal.tensor.barrier join(%[[RESULT_TENSORS]]#0, %[[RESULT_TENSORS]]#1 : tensor<4xf32>, tensor<4xf32>) => %[[SIGNAL]] : !hal.fence
//  CHECK-NEXT:   %[[RET0_VIEW:.+]] = hal.tensor.export %[[READY_TENSORS]]#0 "output0" : tensor<4xf32> -> !hal.buffer_view
//  CHECK-NEXT:   %[[RET1_VIEW:.+]] = hal.tensor.export %[[READY_TENSORS]]#1 "output1" : tensor<4xf32> -> !hal.buffer_view
//  CHECK-NEXT:   util.return %[[RET0_VIEW]], %[[RET1_VIEW]] : !hal.buffer_view, !hal.buffer_view
//  CHECK-NEXT: }

// CHECK-LABEL: util.func private @_asyncEntry(
util.func public @asyncEntry(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %0 = arith.addf %arg0, %arg1 : tensor<4xf32>
  %1 = arith.addf %0, %arg0 : tensor<4xf32>
  util.return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// -----

// CHECK-LABEL: util.func public @bareFunc
//  CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence)
//       CHECK:   util.call @_bareFunc()
//  CHECK-NEXT:   hal.fence.signal<%[[SIGNAL]] : !hal.fence>
//  CHECK-NEXT:   util.return

// CHECK-LABEL: util.func private @_bareFunc(
util.func public @bareFunc() {
  util.return
}

// -----

// CHECK-LABEL: util.func public @primitiveArgOnly
//  CHECK-SAME: (%[[ARG0:.+]]: i32, %[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence)
//  CHECK-NEXT:   util.call @_primitiveArgOnly(%[[ARG0]])
//  CHECK-NEXT:   hal.fence.signal<%[[SIGNAL]] : !hal.fence>
//  CHECK-NEXT:   util.return

// CHECK-LABEL: util.func private @_primitiveArgOnly(
util.func public @primitiveArgOnly(%arg0: i32) {
  %0 = arith.addi %arg0, %arg0 : i32
  util.optimization_barrier %0 : i32
  util.return
}

// -----

// CHECK-LABEL: util.func public @tensorArgOnly
//  CHECK-SAME: (%[[ARG0:.+]]: !hal.buffer_view, %[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence)
//       CHECK:   %[[ARG0_TENSOR:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[ARG0]] "input0" : !hal.buffer_view -> tensor<4xf32>
//  CHECK-NEXT:   util.call @_tensorArgOnly(%[[ARG0_TENSOR]])
//  CHECK-NEXT:   hal.fence.signal<%[[SIGNAL]] : !hal.fence>
//  CHECK-NEXT:   util.return

// CHECK-LABEL: util.func private @_tensorArgOnly(
util.func public @tensorArgOnly(%arg0: tensor<4xf32>) {
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  util.optimization_barrier %0 : tensor<4xf32>
  util.return
}

// -----

// CHECK-LABEL: util.func public @primitiveResultOnly
//  CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence) -> i32
//  CHECK-NEXT:   %[[RESULT:.+]] = util.call @_primitiveResultOnly()
//  CHECK-NEXT:   hal.fence.signal<%[[SIGNAL]] : !hal.fence>
//  CHECK-NEXT:   util.return %[[RESULT]]

// CHECK-LABEL: util.func private @_primitiveResultOnly(
util.func public @primitiveResultOnly() -> i32 {
  %0 = arith.constant 8 : i32
  %1 = util.optimization_barrier %0 : i32
  util.return %1 : i32
}

// -----

// CHECK-LABEL: util.func public @tensorResultOnly
//  CHECK-SAME: (%[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence) -> !hal.buffer_view
//  CHECK-NEXT:   %[[RESULT_TENSOR:.+]] = util.call @_tensorResultOnly()
//  CHECK-NEXT:   %[[READY_TENSOR:.+]] = hal.tensor.barrier join(%[[RESULT_TENSOR]] : tensor<4xf32>) => %[[SIGNAL]] : !hal.fence
//  CHECK-NEXT:   %[[RESULT_VIEW:.+]] = hal.tensor.export %[[READY_TENSOR]]
//  CHECK-NEXT:   util.return %[[RESULT_VIEW]]

// CHECK-LABEL: util.func private @_tensorResultOnly(
util.func public @tensorResultOnly() -> tensor<4xf32> {
  %0 = arith.constant dense<[0.0, 1.0, 2.0, 3.0]> : tensor<4xf32>
  %1 = util.optimization_barrier %0 : tensor<4xf32>
  util.return %1 : tensor<4xf32>
}

// -----

// Tests that imported functions with the coarse-fences execution model
// specified get wrapped with fences. Note that unlike exports controlled by
// compiler flags imports only get the fences when explicitly specified so as
// that is part of their ABI. Users can always manually specify the fences too
// though that's much more verbose.

// CHECK-LABEL: util.func private @import(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view, %arg2: !hal.fence, %arg3: !hal.fence) -> (!hal.buffer_view, !hal.buffer_view)
util.func private @import(tensor<?x2xi32>, tensor<?x3xi32>) -> (tensor<2x?xi32>, tensor<3x?xi32>) attributes {
  iree.abi.model = "coarse-fences",
  nosideeffects
}

// CHECK: util.func private @_import(%[[ARG0_TENSOR:.+]]: tensor<?x2xi32>, %[[ARG1_TENSOR:.+]]: tensor<?x3xi32>) -> (tensor<2x?xi32>, tensor<3x?xi32>) {

// Prepare fences and put a barrier on input arguments:
// CHECK:   %[[DEVICE:.+]] = hal.devices.get %{{.+}}
// CHECK:   %[[WAIT_FENCE:.+]] = hal.fence.create device(%[[DEVICE]]
// CHECK:   %[[ARG_BARRIER:.+]]:2 = hal.tensor.barrier join(%[[ARG0_TENSOR]], %[[ARG1_TENSOR]] : tensor<?x2xi32>, tensor<?x3xi32>) => %[[WAIT_FENCE]] : !hal.fence
// CHECK:   %[[SIGNAL_FENCE:.+]] = hal.fence.create device(%[[DEVICE]]

// Export input arguments to buffer views:
// CHECK:   %[[ARG0_DIM:.+]] = tensor.dim %[[ARG_BARRIER]]#0, %c0
// CHECK:   %[[ARG0_VIEW:.+]] = hal.tensor.export %[[ARG_BARRIER]]#0 : tensor<?x2xi32>{%[[ARG0_DIM]]} -> !hal.buffer_view
// CHECK:   %[[ARG1_DIM:.+]] = tensor.dim %[[ARG_BARRIER]]#1, %c0
// CHECK:   %[[ARG1_VIEW:.+]] = hal.tensor.export %[[ARG_BARRIER]]#1 : tensor<?x3xi32>{%[[ARG1_DIM]]} -> !hal.buffer_view

// Call the import:
// CHECK:   %[[RET_VIEWS:.+]]:2 = util.call @import(%[[ARG0_VIEW]], %[[ARG1_VIEW]], %[[WAIT_FENCE]], %[[SIGNAL_FENCE]]) : (!hal.buffer_view, !hal.buffer_view, !hal.fence, !hal.fence) -> (!hal.buffer_view, !hal.buffer_view)

// Import output results from buffer views:
// CHECK:   %[[RET0_DIM:.+]] = hal.buffer_view.dim<%[[RET_VIEWS]]#0 : !hal.buffer_view>[1]
// CHECK:   %[[RET0_TENSOR:.+]] = hal.tensor.import wait(%[[SIGNAL_FENCE]]) => %[[RET_VIEWS]]#0 : !hal.buffer_view -> tensor<2x?xi32>{%[[RET0_DIM]]}
// CHECK:   %[[RET1_DIM:.+]] = hal.buffer_view.dim<%[[RET_VIEWS]]#1 : !hal.buffer_view>[1]
// CHECK:   %[[RET1_TENSOR:.+]] = hal.tensor.import wait(%[[SIGNAL_FENCE]]) => %[[RET_VIEWS]]#1 : !hal.buffer_view -> tensor<3x?xi32>{%[[RET1_DIM]]}

// CHECK:   util.return %[[RET0_TENSOR]], %[[RET1_TENSOR]] : tensor<2x?xi32>, tensor<3x?xi32>
// CHECK: }

// CHECK: util.func private @caller(%[[ARG0_CALLER:.+]]: tensor<?x2xi32>, %[[ARG1_CALLER:.+]]: tensor<?x3xi32>)
util.func private @caller(%arg0: tensor<?x2xi32>, %arg1: tensor<?x3xi32>) -> (tensor<2x?xi32>, tensor<3x?xi32>) {
  // CHECK: %[[RESULTS:.+]]:2 = util.call @_import(%[[ARG0_CALLER]], %[[ARG1_CALLER]]) : (tensor<?x2xi32>, tensor<?x3xi32>) -> (tensor<2x?xi32>, tensor<3x?xi32>)
  %results:2 = util.call @import(%arg0, %arg1) : (tensor<?x2xi32>, tensor<?x3xi32>) -> (tensor<2x?xi32>, tensor<3x?xi32>)
  // CHECK-NEXT: util.return %[[RESULTS]]#0, %[[RESULTS]]#1
  util.return %results#0, %results#1 : tensor<2x?xi32>, tensor<3x?xi32>
}

// -----

// Tests a side-effect-free import that doesn't take/return reference types.

// CHECK-LABEL: util.func private @importI32(%arg0: i32, %arg1: !hal.fence, %arg2: !hal.fence) -> i32
util.func private @importI32(i32) -> i32 attributes {
  iree.abi.model = "coarse-fences",
  nosideeffects
}

// No fences required as the call has no side-effects and no async resources.
// CHECK: util.func private @_importI32(%[[ARG0:.+]]: i32) -> i32 {
// CHECK:   %[[WAIT_FENCE:.+]] = util.null : !hal.fence
// CHECK:   %[[SIGNAL_FENCE:.+]] = util.null : !hal.fence
// CHECK:   %[[RET0:.+]] = util.call @importI32(%[[ARG0]], %[[WAIT_FENCE]], %[[SIGNAL_FENCE]]) : (i32, !hal.fence, !hal.fence) -> i32
// CHECK:   util.return %[[RET0]] : i32
// CHECK: }

// CHECK: util.func private @callerI32(%[[ARG0_CALLER:.+]]: i32)
util.func private @callerI32(%arg0: i32) -> i32 {
  // CHECK: %[[RESULT:.+]] = util.call @_importI32(%[[ARG0_CALLER]]) : (i32) -> i32
  %result = util.call @importI32(%arg0) : (i32) -> i32
  // CHECK-NEXT: util.return %[[RESULT]]
  util.return %result : i32
}

// -----

// Tests a side-effecting import that requires a host-side wait.

// CHECK-LABEL: util.func private @importI32Effects(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> i32
util.func private @importI32Effects(tensor<4xf32>) -> i32 attributes {
  iree.abi.model = "coarse-fences"
}

// CHECK: util.func private @_importI32Effects(%[[ARG0_TENSOR:.+]]: tensor<4xf32>) -> i32 {

// Wait for the inputs to be ready and create the signal fence to wait on.
// CHECK:   %[[DEVICE:.+]] = hal.devices.get %{{.+}}
// CHECK:   %[[WAIT_FENCE:.+]] = hal.fence.create device(%[[DEVICE]]
// CHECK:   %[[ARG0_BARRIER:.+]] = hal.tensor.barrier join(%[[ARG0_TENSOR]] : tensor<4xf32>) => %[[WAIT_FENCE]] : !hal.fence
// CHECK:   %[[SIGNAL_FENCE:.+]] = hal.fence.create device(%[[DEVICE]]

// Marshal inputs:
// CHECK:   %[[ARG0_VIEW:.+]] = hal.tensor.export %[[ARG0_BARRIER]] : tensor<4xf32> -> !hal.buffer_view

// Make the import call:
// CHECK:   %[[RET0:.+]] = util.call @importI32Effects(%[[ARG0_VIEW]], %[[WAIT_FENCE]], %[[SIGNAL_FENCE]]) : (!hal.buffer_view, !hal.fence, !hal.fence) -> i32

// Perform host-side wait.
// CHECK:   hal.fence.await until([%[[SIGNAL_FENCE]]])

// CHECK:   util.return %[[RET0]] : i32
// CHECK: }

// CHECK: util.func private @callerI32Effects(%[[ARG0_CALLER:.+]]: tensor<4xf32>)
util.func private @callerI32Effects(%arg0: tensor<4xf32>) -> i32 {
  // CHECK: %[[RESULT:.+]] = util.call @_importI32Effects(%[[ARG0_CALLER]]) : (tensor<4xf32>) -> i32
  %result = util.call @importI32Effects(%arg0) : (tensor<4xf32>) -> i32
  // CHECK-NEXT: util.return %[[RESULT]]
  util.return %result : i32
}
