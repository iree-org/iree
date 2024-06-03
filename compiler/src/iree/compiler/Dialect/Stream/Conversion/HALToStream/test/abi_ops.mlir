// RUN: iree-opt --split-input-file --iree-stream-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @importBufferView
// CHECK-SAME: (%[[VIEW:.+]]: !hal.buffer_view)
// CHECK-SAME: -> (!stream.resource<*>, index)
util.func public @importBufferView(%view: !hal.buffer_view) -> tensor<?x?x4xf32> {
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
  // CHECK: util.return %[[RESULT]], %[[SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<?x?x4xf32>
}

// -----

// CHECK-LABEL: @importBufferViewBitcasting
// CHECK-SAME: (%[[VIEW:.+]]: !hal.buffer_view) -> (!stream.resource<*>, index)
util.func public @importBufferViewBitcasting(%view: !hal.buffer_view) -> tensor<4xbf16> {
  //  CHECK-DAG: %[[SIZE:.+]] = stream.tensor.sizeof tensor<4xbf16>
  //      CHECK: %[[RESOURCE:.+]] = stream.tensor.import %[[VIEW]] : !hal.buffer_view ->
  // CHECK-SAME:     tensor<2xui32> in !stream.resource<external>{%[[SIZE]]}
  // CHECK-NEXT: %[[RESULT:.+]] = stream.async.transfer %[[RESOURCE]] :
  // CHECK-SAME:     !stream.resource<external>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %0 = hal.tensor.import %view : !hal.buffer_view -> tensor<2xui32> as tensor<4xbf16>
  // CHECK: util.return %[[RESULT]], %[[SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<4xbf16>
}

// -----

// CHECK-LABEL: @importBufferViewAsync
// CHECK-SAME: (%[[VIEW:.+]]: !hal.buffer_view, %[[FENCE:.+]]: !hal.fence)
// CHECK-SAME: -> (!stream.resource<*>, index)
util.func public @importBufferViewAsync(%view: !hal.buffer_view, %fence: !hal.fence) -> tensor<4xf32> {
  //  CHECK-DAG: %[[SIZE:.+]] = stream.tensor.sizeof tensor<4xf32>
  //      CHECK: %[[ASYNC_RESOURCE:.+]] = stream.tensor.import %[[VIEW]]
  // CHECK-SAME:     : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%[[SIZE]]}
  //      CHECK: %[[TIMEPOINT:.+]] = stream.timepoint.import %[[FENCE]]
  //      CHECK: %[[SYNC_RESOURCE:.+]] = stream.timepoint.await %[[TIMEPOINT]] => %[[ASYNC_RESOURCE]]
  // CHECK-SAME:     : !stream.resource<external>{%[[SIZE]]}
  // CHECK-NEXT: %[[RESULT:.+]] = stream.async.transfer %[[SYNC_RESOURCE]]
  // CHECK-SAME:     : !stream.resource<external>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %0 = hal.tensor.import wait(%fence) => %view : !hal.buffer_view -> tensor<4xf32>
  // CHECK: util.return %[[RESULT]], %[[SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @exportBufferView
// CHECK-SAME: (%[[TENSOR:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index)
util.func public @exportBufferView(%tensor: tensor<?x?x4xf32>, %dim0: index, %dim1: index) -> !hal.buffer_view {
  //      CHECK: %[[VIEW:.+]] = stream.async.transfer %[[TENSOR]] :
  // CHECK-SAME:     !stream.resource<*>{%[[SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  // CHECK-NEXT: %[[RESULT:.+]] = stream.tensor.export %[[VIEW]] :
  // CHECK-SAME:     tensor<?x?x4xf32>{%[[DIM0]], %[[DIM1]]} in !stream.resource<external>{%[[SIZE]]}
  // CHECK-SAME:     -> !hal.buffer_view
  %0 = hal.tensor.export %tensor : tensor<?x?x4xf32>{%dim0, %dim1} -> !hal.buffer_view
  // CHECK: util.return %[[RESULT]]
  util.return %0 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @aliasStorage
// CHECK-SAME: (%[[TENSOR:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[DIM0:.+]]: index, %[[STORAGE:.+]]: !hal.buffer)
util.func public @aliasStorage(%tensor: tensor<?x4xf32>, %dim0: index, %storage: !hal.buffer) -> tensor<?x4xf32> {
  // CHECK: %[[MIN_STORAGE_SIZE:.+]] = stream.tensor.sizeof tensor<?x4xf32>{%[[DIM0]]}
  // CHECK: %[[STORAGE_RESOURCE:.+]] = stream.tensor.import %[[STORAGE]] : !hal.buffer -> tensor<?x4xf32>{%[[DIM0]]} in !stream.resource<external>{%[[MIN_STORAGE_SIZE]]}
  // CHECK: %[[UPDATE:.+]] = stream.async.update %[[TENSOR]], %[[STORAGE_RESOURCE]][%c0 to %[[SIZE]]] : !stream.resource<*>{%[[SIZE]]} -> %[[STORAGE_RESOURCE]] as !stream.resource<external>{%[[MIN_STORAGE_SIZE]]}
  // CHECK: %[[SLICE:.+]] = stream.async.slice %[[UPDATE]][%c0 to %[[SIZE]]] : !stream.resource<external>{%[[MIN_STORAGE_SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  // CHECK: %[[RESULT:.+]] = stream.async.transfer %[[SLICE]] : !stream.resource<external>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %0 = hal.tensor.alias %tensor : tensor<?x4xf32>{%dim0} to %storage : !hal.buffer
  // CHECK: util.return %[[RESULT]]
  util.return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @aliasStorageAsync
// CHECK-SAME: (%[[TENSOR:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[DIM0:.+]]: index, %[[STORAGE:.+]]: !hal.buffer, %[[FENCE:.+]]: !hal.fence)
util.func public @aliasStorageAsync(%tensor: tensor<?x4xf32>, %dim0: index, %storage: !hal.buffer, %fence: !hal.fence) -> tensor<?x4xf32> {
  // CHECK-DAG: %[[MIN_STORAGE_SIZE:.+]] = stream.tensor.sizeof tensor<?x4xf32>{%[[DIM0]]}
  // CHECK-DAG: %[[UNREADY_STORAGE:.+]] = stream.tensor.import %[[STORAGE]] : !hal.buffer -> tensor<?x4xf32>{%[[DIM0]]} in !stream.resource<external>{%[[MIN_STORAGE_SIZE]]}
  // CHECK-DAG: %[[TIMEPOINT:.+]] = stream.timepoint.import %[[FENCE]]
  // CHECK-DAG: %[[READY_STORAGE:.+]] = stream.timepoint.await %[[TIMEPOINT]] => %[[UNREADY_STORAGE]] : !stream.resource<external>{%[[MIN_STORAGE_SIZE]]}
  // CHECK: %[[UPDATE:.+]] = stream.async.update %[[TENSOR]], %[[READY_STORAGE]][%c0 to %[[SIZE]]] : !stream.resource<*>{%[[SIZE]]} -> %[[READY_STORAGE]] as !stream.resource<external>{%[[MIN_STORAGE_SIZE]]}
  // CHECK: %[[SLICE:.+]] = stream.async.slice %[[UPDATE]][%c0 to %[[SIZE]]] : !stream.resource<external>{%[[MIN_STORAGE_SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  // CHECK: %[[RESULT:.+]] = stream.async.transfer %[[SLICE]] : !stream.resource<external>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %0 = hal.tensor.alias wait(%fence) => %tensor : tensor<?x4xf32>{%dim0} to %storage : !hal.buffer
  // CHECK: util.return %[[RESULT]]
  util.return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @tensorBarrier
// CHECK-SAME: (%[[TENSOR0:.+]]: !stream.resource<*>, %[[SIZE0:.+]]: index, %[[TENSOR1:.+]]: !stream.resource<*>, %[[SIZE1:.+]]: index, %[[FENCE:.+]]: !hal.fence)
util.func public @tensorBarrier(%tensor0: tensor<3xf32>, %tensor1: tensor<?xf32>, %fence: !hal.fence) -> (tensor<3xf32>, tensor<?xf32>) {
  //  CHECK-DAG: %[[TENSOR0_AFTER:.+]], %[[TENSOR0_BARRIER:.+]] = stream.timepoint.barrier %[[TENSOR0]] : !stream.resource<*>{%[[SIZE0]]} => !stream.timepoint
  //  CHECK-DAG: %[[TENSOR1_AFTER:.+]], %[[TENSOR1_BARRIER:.+]] = stream.timepoint.barrier %[[TENSOR1]] : !stream.resource<*>{%[[SIZE1]]} => !stream.timepoint
  // CHECK-NEXT: %[[JOIN:.+]] = stream.timepoint.join max(%[[TENSOR0_BARRIER]], %[[TENSOR1_BARRIER]]) => !stream.timepoint
  // CHECK-NEXT: stream.timepoint.chain_external %[[JOIN]] => (%[[FENCE]] : !hal.fence)
  %0:2 = hal.tensor.barrier join(%tensor0, %tensor1 : tensor<3xf32>, tensor<?xf32>) => %fence : !hal.fence
  // CHECK: util.return %[[TENSOR0_AFTER]], %[[SIZE0]], %[[TENSOR1_AFTER]], %[[SIZE1]]
  util.return %0#0, %0#1 : tensor<3xf32>, tensor<?xf32>
}
