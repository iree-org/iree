// RUN: iree-opt -iree-abi-wrap-entry-points -split-input-file %s | IreeFileCheck %s

// CHECK-LABEL: func @dynamicEntry(
//  CHECK-SAME:   %[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view
//  CHECK-SAME: -> (
//  CHECK-SAME:   !hal.buffer_view, !hal.buffer_view
//  CHECK-SAME: ) attributes {
//  CHECK-SAME:   iree.abi.stub
//  CHECK-SAME: } {
//  CHECK-NEXT:   %[[ARG0_DIM0:.+]] = hal.buffer_view.dim<%[[ARG0]] : !hal.buffer_view>[0] : index
//  CHECK-NEXT:   %[[ARG0_TENSOR:.+]] = hal.tensor.cast %[[ARG0]] : !hal.buffer_view -> tensor<?x8x8x3xf32>{%[[ARG0_DIM0]]}
//  CHECK-NEXT:   %[[ARG1_DIM0:.+]] = hal.buffer_view.dim<%[[ARG1]] : !hal.buffer_view>[0] : index
//  CHECK-NEXT:   %[[ARG1_TENSOR:.+]] = hal.tensor.cast %[[ARG1]] : !hal.buffer_view -> tensor<?x8x8x3xf32>{%[[ARG1_DIM0]]}
//  CHECK-NEXT:   %[[RET_TENSOR:.+]]:2 = call @_dynamicEntry(%[[ARG0_TENSOR]], %[[ARG1_TENSOR]])
//       CHECK:   %[[RET0_DIM0:.+]] = tensor.dim %[[RET_TENSOR]]#0, %c0{{.*}} : tensor<?x8x8x3xf32>
//  CHECK-NEXT:   %[[RET0_VIEW:.+]] = hal.tensor.cast %[[RET_TENSOR]]#0 : tensor<?x8x8x3xf32>{%[[RET0_DIM0]]} -> !hal.buffer_view
//       CHECK:   %[[RET1_DIM0:.+]] = tensor.dim %[[RET_TENSOR]]#1, %c0{{.*}} : tensor<?x8x8x3xf32>
//  CHECK-NEXT:   %[[RET1_VIEW:.+]] = hal.tensor.cast %[[RET_TENSOR]]#1 : tensor<?x8x8x3xf32>{%[[RET1_DIM0]]} -> !hal.buffer_view
//  CHECK-NEXT:   return %[[RET0_VIEW]], %[[RET1_VIEW]] : !hal.buffer_view, !hal.buffer_view
//  CHECK-NEXT: }

// CHECK-LABEL: func private @_dynamicEntry(
func @dynamicEntry(%arg0: tensor<?x8x8x3xf32>, %arg1: tensor<?x8x8x3xf32>) ->
    (tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>) {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>) -> tensor<?x8x8x3xf32>
  %1 = "mhlo.add"(%0, %arg0) : (tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>) -> tensor<?x8x8x3xf32>
  return %0, %1 : tensor<?x8x8x3xf32>, tensor<?x8x8x3xf32>
}

// -----

// CHECK-LABEL: func @wrappedAlready
//  CHECK-SAME: (%arg0: !hal.buffer_view) -> !hal.buffer_view
//  CHECK-SAME: attributes {iree.abi.stub}
func @wrappedAlready(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
  return %arg0 : !hal.buffer_view
}
// CHECK-NOT: func @_wrappedAlready
