// RUN: iree-opt --pass-pipeline='builtin.module(iree-abi-wrap-entry-points{invocation-model=coarse-fences})' --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @asyncEntry(
//  CHECK-SAME:   %[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view, %[[WAIT:.+]]: !hal.fence, %[[SIGNAL:.+]]: !hal.fence
//  CHECK-SAME: -> (
//  CHECK-SAME:   !hal.buffer_view, !hal.buffer_view
//  CHECK-SAME: ) attributes {
//  CHECK-SAME:   iree.abi.stub
//  CHECK-SAME:   iree.reflection = {iree.abi.model = "coarse-fences"}
//  CHECK-SAME: } {
//  CHECK-NEXT:   %[[ARG0_TENSOR:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[ARG0]] : !hal.buffer_view -> tensor<4xf32>
//  CHECK-NEXT:   %[[ARG1_TENSOR:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[ARG1]] : !hal.buffer_view -> tensor<4xf32>
//  CHECK-NEXT:   %[[RESULT_TENSORS:.+]]:2 = call @_asyncEntry(%[[ARG0_TENSOR]], %[[ARG1_TENSOR]])
//  CHECK-NEXT:   %[[READY_TENSORS:.+]]:2 = hal.tensor.barrier join(%[[RESULT_TENSORS]]#0, %[[RESULT_TENSORS]]#1 : tensor<4xf32>, tensor<4xf32>) => %[[SIGNAL]] : !hal.fence
//  CHECK-NEXT:   %[[RET0_VIEW:.+]] = hal.tensor.export %[[READY_TENSORS]]#0 : tensor<4xf32> -> !hal.buffer_view
//  CHECK-NEXT:   %[[RET1_VIEW:.+]] = hal.tensor.export %[[READY_TENSORS]]#1 : tensor<4xf32> -> !hal.buffer_view
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
//       CHECK:   %[[ARG0_TENSOR:.+]] = hal.tensor.import wait(%[[WAIT]]) => %[[ARG0]] : !hal.buffer_view -> tensor<4xf32>
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

