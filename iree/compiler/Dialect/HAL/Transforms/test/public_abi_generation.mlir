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
