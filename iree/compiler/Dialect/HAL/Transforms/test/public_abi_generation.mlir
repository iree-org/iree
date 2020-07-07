// RUN: iree-opt -split-input-file -iree-hal-public-abi-generation %s | IreeFileCheck %s

// CHECK-LABEL: @noReflectionExport
// CHECK-SAME: attributes {iree.module.export}
func @noReflectionExport(%arg0 : tensor<4xf32>) -> tensor<4xf32>
    attributes {iree.module.export} {
  return %arg0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @staticTwoArg
// Note: reflection matches signature:
//   (%arg0 : tensor<4x4xi64>, %arg1 : tensor<5x6xi64>) -> tensor<5x6xi64>
// Original func should be rewritten to export with $raw suffix with no
// reflection metadata.
// CHECK-SAME: {iree.module.export = "staticTwoArg$raw"}
// A new function with $sync suffix based on buffer_view should be generated.
// CHECK: func @staticTwoArg$sync(%[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view)
// CHECK-SAME: attributes
// CHECK-SAME:   iree.abi.stub
// CHECK-SAME:   iree.module.export = "staticTwoArg"
// CHECK-SAME:   iree.reflection = {f = "I19!B7!t7d4d4B7!t7d5d6R10!B7!t7d5d6", fv = "1"}
func @staticTwoArg(%arg0 : !hal.buffer, %arg1 : !hal.buffer) -> !hal.buffer
    attributes {iree.module.export,
      iree.reflection = {f = "I19!B7!t7d4d4B7!t7d5d6R10!B7!t7d5d6", fv = "1"}}
{
  // CHECK-DAG: %[[BUFFER0:.+]] = hal.buffer_view.buffer %[[ARG0]] : !hal.buffer
  // CHECK-DAG: %[[BUFFER1:.+]] = hal.buffer_view.buffer %[[ARG1]] : !hal.buffer
  // CHECK-DAG: %[[R0:.+]] = call @staticTwoArg(%[[BUFFER0]], %[[BUFFER1]])
  // CHECK-DAG: %[[C5:.+]] = constant 5 : index
  // CHECK-DAG: %[[C6:.+]] = constant 6 : index
  // CHECK-DAG: %[[VIEW:.+]] = hal.buffer_view.create %[[R0]], shape = [%[[C5]], %[[C6]]], element_type = 16777280 : !hal.buffer_view
  // CHECK: return %[[VIEW]]
  return %arg1 : !hal.buffer
}
// A new function with $async suffix based on buffer_view with wait and signal
// semaphore arguments should be generated. For now, it should just wrap $sync.
// CHECK: func @staticTwoArg$async(%[[ARG0:.+]]: !hal.semaphore, %[[ARG1:.+]]: index, %[[ARG2:.+]]: !hal.buffer_view, %[[ARG3:.+]]: !hal.buffer_view, %[[ARG4:.+]]: !hal.semaphore, %[[ARG5:.+]]: index)
// CHECK: %[[WAITRESULT:.+]] = hal.semaphore.await %[[ARG0]], min_value = %[[ARG1]] : i32
// CHECK: hal.check_success %[[WAITRESULT]]
// CHECK: %[[RESULT:.+]] = call @staticTwoArg$sync(%[[ARG2]], %[[ARG3]]) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view
// CHECK: hal.semaphore.signal %[[ARG4]], value = %[[ARG5]]
// CHECK: return %[[RESULT]] : !hal.buffer_view

// -----
// CHECK-LABEL: @dynamicTwoDims
// Note: reflection matches signature:
//   (%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32>
// Original func should be rewritten to export with $raw suffix with no
// reflection metadata.
// CHECK-SAME: {iree.module.export = "dynamicTwoDims$raw"}
// A new function with $sync suffix based on buffer_view should be generated.
// CHECK: func @dynamicTwoDims$sync(%[[ARG0:.+]]: !hal.buffer_view)
// CHECK-SAME: attributes
// CHECK-SAME:   iree.abi.stub
// CHECK-SAME:   iree.module.export = "dynamicTwoDims"
// CHECK-SAME:   iree.reflection = {f = "I10!B7!d-1d-1R10!B7!d-1d-1", fv = "1"}
// CHECK-DAG: %[[BUFFER:.+]] = hal.buffer_view.buffer %[[ARG0]] : !hal.buffer
// CHECK-DAG: %[[DIM0:.+]] = hal.buffer_view.dim %[[ARG0]], 0 : index
// CHECK-DAG: %[[DIM1:.+]] = hal.buffer_view.dim %[[ARG0]], 1 : index
// CHECK-DAG: %[[RESULT:.+]]:3 = call @dynamicTwoDims(%[[BUFFER]], %[[DIM0]], %[[DIM1]])
// CHECK-DAG: %[[RESULT_VIEW:.+]] = hal.buffer_view.create %[[RESULT]]#0, shape = [%[[RESULT]]#1, %[[RESULT]]#2], element_type = 50331680 : !hal.buffer_view
// CHECK: return %[[RESULT_VIEW]]
// A new function with $async suffix based on buffer_view with wait and signal
// semaphore arguments should be generated. For now, it should just wrap $sync.
// CHECK: func @dynamicTwoDims$async(%[[ARG0:.+]]: !hal.semaphore, %[[ARG1:.+]]: index, %[[ARG2:.+]]: !hal.buffer_view, %[[ARG3:.+]]: !hal.semaphore, %[[ARG4:.+]]: index)
// CHECK: %[[WAITRESULT:.+]] = hal.semaphore.await %[[ARG0]], min_value = %[[ARG1]] : i32
// CHECK: hal.check_success %[[WAITRESULT]]
// CHECK: %[[RESULT:.+]] = call @dynamicTwoDims$sync(%[[ARG2]]) : (!hal.buffer_view) -> !hal.buffer_view
// CHECK: hal.semaphore.signal %[[ARG3]], value = %[[ARG4]]
// CHECK: return %[[RESULT]] : !hal.buffer_view
func @dynamicTwoDims(%arg0 : !hal.buffer, %arg1 : index, %arg2 : index) -> (!hal.buffer, index, index)
    attributes {iree.module.export,
      iree.reflection = {f = "I10!B7!d-1d-1R10!B7!d-1d-1", fv = "1"}}
{
  %0 = constant 5 : index
  %1 = constant 6 : index
  return %arg0, %0, %1 : !hal.buffer, index, index
}
