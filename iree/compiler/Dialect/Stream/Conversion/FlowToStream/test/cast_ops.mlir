// RUN: iree-opt -split-input-file -iree-stream-conversion -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: @importBufferView
// CHECK-SAME: (%[[VIEW:.+]]: !hal.buffer_view)
// CHECK-SAME: -> (!stream.resource<*>, index)
func @importBufferView(%view: !hal.buffer_view) -> tensor<?x?x4xf32> {
  //  CHECK-DAG: %[[DIM0:.+]] = hal.buffer_view.dim{{.+}}[0]
  %dim0 = hal.buffer_view.dim<%view : !hal.buffer_view>[0] : index
  //  CHECK-DAG: %[[DIM1:.+]] = hal.buffer_view.dim{{.+}}[1]
  %dim1 = hal.buffer_view.dim<%view : !hal.buffer_view>[1] : index
  //  CHECK-DAG: %[[SIZE:.+]] = stream.tensor.sizeof tensor<?x?x4xf32>{%[[DIM0]], %[[DIM1]]} : index
  //      CHECK: %[[RESOURCE:.+]] = stream.tensor.import %[[VIEW]] : !hal.buffer_view ->
  // CHECK-SAME:     tensor<?x?x4xf32>{%[[DIM0]], %[[DIM1]]} in !stream.resource<external>{%[[SIZE]]}
  // CHECK-NEXT: %[[RESULT:.+]] = stream.async.transfer %[[RESOURCE]] :
  // CHECK-SAME:     !stream.resource<external>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %0 = hal.tensor.cast %view : !hal.buffer_view -> tensor<?x?x4xf32>{%dim0, %dim1}
  // CHECK: return %[[RESULT]], %[[SIZE]] : !stream.resource<*>, index
  return %0 : tensor<?x?x4xf32>
}

// -----

// CHECK-LABEL: @exportBufferView
// CHECK-SAME: (%[[TENSOR:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index)
func @exportBufferView(%tensor: tensor<?x?x4xf32>, %dim0: index, %dim1: index) -> !hal.buffer_view {
  //      CHECK: %[[VIEW:.+]] = stream.async.transfer %[[TENSOR]] :
  // CHECK-SAME:     !stream.resource<*>{%[[SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  // CHECK-NEXT: %[[RESULT:.+]] = stream.tensor.export %[[VIEW]] :
  // CHECK-SAME:     tensor<?x?x4xf32>{%[[DIM0]], %[[DIM1]]} in !stream.resource<external>{%[[SIZE]]}
  // CHECK-SAME:     -> !hal.buffer_view
  %0 = hal.tensor.cast %tensor : tensor<?x?x4xf32>{%dim0, %dim1} -> !hal.buffer_view
  // CHECK: return %[[RESULT]]
  return %0 : !hal.buffer_view
}
