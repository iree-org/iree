// RUN: iree-opt --split-input-file --iree-stream-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @importBufferView
// CHECK-SAME: (%[[VIEW:.+]]: !hal.buffer_view)
// CHECK-SAME: -> (!stream.resource<*>, index)
func.func @importBufferView(%view: !hal.buffer_view) -> tensor<?x?x4xf32> {
  //  CHECK-DAG: %[[DIM0:.+]] = hal.buffer_view.dim{{.+}}[0]
  %dim0 = hal.buffer_view.dim<%view : !hal.buffer_view>[0] : index
  //  CHECK-DAG: %[[DIM1:.+]] = hal.buffer_view.dim{{.+}}[1]
  %dim1 = hal.buffer_view.dim<%view : !hal.buffer_view>[1] : index
  //  CHECK-DAG: %[[SIZE:.+]] = stream.tensor.sizeof tensor<?x?x4xf32>{%[[DIM0]], %[[DIM1]]} : index
  //      CHECK: %[[RESOURCE:.+]] = stream.tensor.import %[[VIEW]] : !hal.buffer_view ->
  // CHECK-SAME:     tensor<?x?x4xf32>{%[[DIM0]], %[[DIM1]]} in !stream.resource<external>{%[[SIZE]]}
  // CHECK-NEXT: %[[RESULT:.+]] = stream.async.transfer %[[RESOURCE]] :
  // CHECK-SAME:     !stream.resource<external>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %0 = hal.tensor.import %view : !hal.buffer_view -> tensor<?x?x4xf32>{%dim0, %dim1}
  // CHECK: return %[[RESULT]], %[[SIZE]] : !stream.resource<*>, index
  return %0 : tensor<?x?x4xf32>
}

// -----

// CHECK-LABEL: @importBufferViewBitcasting
// CHECK-SAME: (%[[VIEW:.+]]: !hal.buffer_view) -> (!stream.resource<*>, index)
func.func @importBufferViewBitcasting(%view: !hal.buffer_view) -> tensor<4xbf16> {
  //  CHECK-DAG: %[[SIZE:.+]] = stream.tensor.sizeof tensor<4xbf16>
  //      CHECK: %[[RESOURCE:.+]] = stream.tensor.import %[[VIEW]] : !hal.buffer_view ->
  // CHECK-SAME:     tensor<2xui32> in !stream.resource<external>{%[[SIZE]]}
  // CHECK-NEXT: %[[RESULT:.+]] = stream.async.transfer %[[RESOURCE]] :
  // CHECK-SAME:     !stream.resource<external>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %0 = hal.tensor.import %view : !hal.buffer_view -> tensor<2xui32> as tensor<4xbf16>
  // CHECK: return %[[RESULT]], %[[SIZE]] : !stream.resource<*>, index
  return %0 : tensor<4xbf16>
}

// -----

// CHECK-LABEL: @importBufferViewAsync
// CHECK-SAME: (%[[VIEW:.+]]: !hal.buffer_view, %[[FENCE:.+]]: !hal.fence)
// CHECK-SAME: -> (!stream.resource<*>, index)
func.func @importBufferViewAsync(%view: !hal.buffer_view, %fence: !hal.fence) -> tensor<4xf32> {
  //  CHECK-DAG: %[[SIZE:.+]] = stream.tensor.sizeof tensor<4xf32>
  //      CHECK: %[[ASYNC_RESOURCE:.+]] = stream.tensor.import %[[VIEW]]
  // CHECK-SAME:     : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%[[SIZE]]}
  //      CHECK: %[[TIMEPOINT:.+]] = stream.timepoint.import %[[FENCE]]
  //      CHECK: %[[SYNC_RESOURCE:.+]] = stream.timepoint.await %[[TIMEPOINT]] => %[[ASYNC_RESOURCE]]
  // CHECK-SAME:     : !stream.resource<external>{%[[SIZE]]}
  // CHECK-NEXT: %[[RESULT:.+]] = stream.async.transfer %[[SYNC_RESOURCE]]
  // CHECK-SAME:     : !stream.resource<external>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %0 = hal.tensor.import wait(%fence) => %view : !hal.buffer_view -> tensor<4xf32>
  // CHECK: return %[[RESULT]], %[[SIZE]] : !stream.resource<*>, index
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @exportBufferView
// CHECK-SAME: (%[[TENSOR:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index)
func.func @exportBufferView(%tensor: tensor<?x?x4xf32>, %dim0: index, %dim1: index) -> !hal.buffer_view {
  //      CHECK: %[[VIEW:.+]] = stream.async.transfer %[[TENSOR]] :
  // CHECK-SAME:     !stream.resource<*>{%[[SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  // CHECK-NEXT: %[[RESULT:.+]] = stream.tensor.export %[[VIEW]] :
  // CHECK-SAME:     tensor<?x?x4xf32>{%[[DIM0]], %[[DIM1]]} in !stream.resource<external>{%[[SIZE]]}
  // CHECK-SAME:     -> !hal.buffer_view
  %0 = hal.tensor.export %tensor : tensor<?x?x4xf32>{%dim0, %dim1} -> !hal.buffer_view
  // CHECK: return %[[RESULT]]
  return %0 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @exportBufferViewInPlace
// CHECK-SAME: (%[[TENSOR:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index, %[[STORAGE:.+]]: !hal.buffer)
func.func @exportBufferViewInPlace(%tensor: tensor<?x?x4xf32>, %dim0: index, %dim1: index, %storage: !hal.buffer) -> !hal.buffer_view {
  //      CHECK: %[[STORAGE_LENGTH:.+]] = hal.buffer.length<%[[STORAGE]]
  // CHECK-NEXT: %[[STORAGE_IMPORT:.+]] = stream.tensor.import %[[STORAGE]]
  // CHECK-SAME:   : !hal.buffer -> tensor<?x?x4xf32>{%[[DIM0]], %[[DIM1]]} in !stream.resource<external>{%[[STORAGE_LENGTH]]}
  // CHECK-NEXT: %[[STORAGE_UPDATE:.+]] = stream.async.update %[[TENSOR]], %[[STORAGE_IMPORT]][%c0 to %[[SIZE]]]
  // CHECK-SAME:   : !stream.resource<*>{%[[SIZE]]} -> %[[STORAGE_IMPORT]] as !stream.resource<external>{%[[STORAGE_LENGTH]]}
  // CHECK-NEXT: %[[STORAGE_RESULT:.+]] = stream.tensor.export %[[STORAGE_UPDATE]] :
  // CHECK-SAME:     tensor<?x?x4xf32>{%[[DIM0]], %[[DIM1]]} in !stream.resource<external>{%[[STORAGE_LENGTH]]}
  // CHECK-SAME:     -> !hal.buffer_view
  %0 = hal.tensor.export %tensor into(%storage : !hal.buffer) : tensor<?x?x4xf32>{%dim0, %dim1} -> !hal.buffer_view
  // CHECK: return %[[STORAGE_RESULT]]
  return %0 : !hal.buffer_view
}
