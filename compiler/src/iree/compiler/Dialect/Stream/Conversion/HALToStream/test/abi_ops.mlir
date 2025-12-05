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
  // CHECK-NEXT: %[[RESULT:.+]] = stream.async.cast %[[RESOURCE]] :
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
  // CHECK-NEXT: %[[RESULT:.+]] = stream.async.cast %[[RESOURCE]] :
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
  // CHECK-NEXT: %[[RESULT:.+]] = stream.async.cast %[[SYNC_RESOURCE]]
  // CHECK-SAME:     : !stream.resource<external>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %0 = hal.tensor.import wait(%fence) => %view : !hal.buffer_view -> tensor<4xf32>
  // CHECK: util.return %[[RESULT]], %[[SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @exportBufferView
// CHECK-SAME: (%[[TENSOR:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[DIM0:.+]]: index, %[[DIM1:.+]]: index)
util.func public @exportBufferView(%tensor: tensor<?x?x4xf32>, %dim0: index, %dim1: index) -> !hal.buffer_view {
  //      CHECK: %[[VIEW:.+]] = stream.async.cast %[[TENSOR]] :
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
  // CHECK: %[[RESULT:.+]] = stream.async.clone %[[SLICE]] : !stream.resource<external>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
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
  // CHECK: %[[RESULT:.+]] = stream.async.clone %[[SLICE]] : !stream.resource<external>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
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

// -----

// Tests import on dev_a while function affinity is dev_b.
// A clone happens on dev_a (same-affinity lifetime change) but no transfer is inserted yet.

// CHECK-LABEL: @importBufferViewCrossDevice
// CHECK-SAME: (%[[VIEW:.+]]: !hal.buffer_view)
// CHECK-SAME: -> (!stream.resource<*>, index)
util.func public @importBufferViewCrossDevice(%view: !hal.buffer_view) -> tensor<4xf32> attributes {
  stream.affinity = #hal.device.promise<@dev_b>
} {
  //  CHECK-DAG: %[[SIZE:.+]] = stream.tensor.sizeof on(#hal.device.promise<@dev_a>) tensor<4xf32>
  //      CHECK: %[[RESOURCE:.+]] = stream.tensor.import on(#hal.device.promise<@dev_a>) %[[VIEW]] : !hal.buffer_view ->
  // CHECK-SAME:     tensor<4xf32> in !stream.resource<external>{%[[SIZE]]}
  // CHECK-NEXT: %[[CLONE:.+]] = stream.async.cast on(#hal.device.promise<@dev_a>) %[[RESOURCE]] :
  // CHECK-SAME:     !stream.resource<external>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %0 = hal.tensor.import on(#hal.device.promise<@dev_a>) %view : !hal.buffer_view -> tensor<4xf32>
  // CHECK: util.return %[[CLONE]], %[[SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<4xf32>
}

// -----

// Tests export on dev_b while function affinity is dev_a.
// A clone happens implicitly without affinity.

// CHECK-LABEL: @exportBufferViewCrossDevice
// CHECK-SAME: (%[[TENSOR:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index)
util.func public @exportBufferViewCrossDevice(%tensor: tensor<4xf32>) -> !hal.buffer_view attributes {
  stream.affinity = #hal.device.promise<@dev_a>
} {
  //      CHECK: %[[CLONE:.+]] = stream.async.cast %[[TENSOR]] :
  // CHECK-SAME:     !stream.resource<*>{%[[SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  // CHECK-NEXT: %[[VIEW:.+]] = stream.tensor.export %[[CLONE]] :
  // CHECK-SAME:     tensor<4xf32> in !stream.resource<external>{%[[SIZE]]}
  // CHECK-SAME:     -> !hal.buffer_view
  %0 = hal.tensor.export on(#hal.device.promise<@dev_b>) %tensor : tensor<4xf32> -> !hal.buffer_view
  // CHECK: util.return %[[VIEW]]
  util.return %0 : !hal.buffer_view
}

// -----

// Tests aliasing storage on dev_b while function affinity is dev_a.
// This generates a transfer from dev_a to dev_b, operations on dev_b, but no transfer back.

// CHECK-LABEL: @aliasStorageCrossDevice
// CHECK-SAME: (%[[TENSOR:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[STORAGE:.+]]: !hal.buffer)
// CHECK-SAME: -> (!stream.resource<*>, index)
util.func public @aliasStorageCrossDevice(%tensor: tensor<4xf32>, %storage: !hal.buffer) -> tensor<4xf32> attributes {
  stream.affinity = #hal.device.promise<@dev_a>
} {
  //      CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[TENSOR]] : !stream.resource<*>{%[[SIZE]]}
  // CHECK-SAME:     -> to(#hal.device.promise<@dev_b>) !stream.resource<*>{%[[SIZE]]}
  //  CHECK-DAG: %[[STORAGE_SIZE:.+]] = stream.tensor.sizeof on(#hal.device.promise<@dev_b>) tensor<4xf32>
  //      CHECK: %[[STORAGE_RESOURCE:.+]] = stream.tensor.import on(#hal.device.promise<@dev_b>) %[[STORAGE]] :
  // CHECK-SAME:     !hal.buffer -> tensor<4xf32> in !stream.resource<external>{%[[STORAGE_SIZE]]}
  //      CHECK: %[[UPDATE:.+]] = stream.async.update on(#hal.device.promise<@dev_b>) %[[TRANSFER]], %[[STORAGE_RESOURCE]][%c0 to %[[SIZE]]] :
  // CHECK-SAME:     !stream.resource<*>{%[[SIZE]]} -> %[[STORAGE_RESOURCE]] as !stream.resource<external>{%[[STORAGE_SIZE]]}
  //      CHECK: %[[SLICE:.+]] = stream.async.slice on(#hal.device.promise<@dev_b>) %[[UPDATE]][%c0 to %[[SIZE]]] :
  // CHECK-SAME:     !stream.resource<external>{%[[STORAGE_SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  //      CHECK: %[[CLONE:.+]] = stream.async.clone on(#hal.device.promise<@dev_b>) %[[SLICE]] :
  // CHECK-SAME:     !stream.resource<external>{%[[SIZE]]} -> !stream.resource<*>{%[[SIZE]]}
  %0 = hal.tensor.alias on(#hal.device.promise<@dev_b>) %tensor : tensor<4xf32> to %storage : !hal.buffer
  // CHECK: util.return %[[CLONE]], %[[SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @tensorTransients
// CHECK-SAME: (%[[TENSOR:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[STORAGE:.+]]: !hal.buffer)
// CHECK-SAME: -> (!stream.resource<*>, index)
util.func public @tensorTransients(%tensor: tensor<4xf32>, %storage: !hal.buffer) -> tensor<4xf32> {
  // CHECK: %[[BARRIER_RESULT:.+]], %[[BARRIER_TP:.+]] = stream.timepoint.barrier %[[TENSOR]] : !stream.resource<*>{%[[SIZE]]} => !stream.timepoint
  // CHECK: %[[BUFFER_LENGTH:.+]] = hal.buffer.length<%[[STORAGE]] : !hal.buffer> : index
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import %[[STORAGE]] : !hal.buffer -> tensor<?xi8>{%[[BUFFER_LENGTH]]} in !stream.resource<transient>{%[[BUFFER_LENGTH]]}
  // CHECK: %[[TRANSIENTS_RESULT:.+]], %[[TRANSIENTS_TP:.+]] = stream.resource.transients await(%[[BARRIER_TP]]) => %[[BARRIER_RESULT]] : !stream.resource<*>{%[[SIZE]]} from %[[IMPORTED]] : !stream.resource<transient>{%[[BUFFER_LENGTH]]} => !stream.timepoint
  // CHECK: %[[RESULT:.+]] = stream.timepoint.await %[[TRANSIENTS_TP]] => %[[TRANSIENTS_RESULT]] : !stream.resource<*>{%[[SIZE]]}
  %0 = hal.tensor.transients %tensor : tensor<4xf32> from %storage : !hal.buffer
  // CHECK: util.return %[[RESULT]], %[[SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @tensorTransientsDynamic
// CHECK-SAME: (%[[TENSOR:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[DIM:.+]]: index, %[[STORAGE:.+]]: !hal.buffer)
// CHECK-SAME: -> (!stream.resource<*>, index)
util.func public @tensorTransientsDynamic(%tensor: tensor<?x4xf32>, %dim: index, %storage: !hal.buffer) -> tensor<?x4xf32> {
  // CHECK: %[[BARRIER_RESULT:.+]], %[[BARRIER_TP:.+]] = stream.timepoint.barrier %[[TENSOR]] : !stream.resource<*>{%[[SIZE]]} => !stream.timepoint
  // CHECK: %[[BUFFER_LENGTH:.+]] = hal.buffer.length<%[[STORAGE]] : !hal.buffer> : index
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import %[[STORAGE]] : !hal.buffer -> tensor<?xi8>{%[[BUFFER_LENGTH]]} in !stream.resource<transient>{%[[BUFFER_LENGTH]]}
  // CHECK: %[[TRANSIENTS_RESULT:.+]], %[[TRANSIENTS_TP:.+]] = stream.resource.transients await(%[[BARRIER_TP]]) => %[[BARRIER_RESULT]] : !stream.resource<*>{%[[SIZE]]} from %[[IMPORTED]] : !stream.resource<transient>{%[[BUFFER_LENGTH]]} => !stream.timepoint
  // CHECK: %[[RESULT:.+]] = stream.timepoint.await %[[TRANSIENTS_TP]] => %[[TRANSIENTS_RESULT]] : !stream.resource<*>{%[[SIZE]]}
  %0 = hal.tensor.transients %tensor : tensor<?x4xf32>{%dim} from %storage : !hal.buffer
  // CHECK: util.return %[[RESULT]], %[[SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @tensorTransientsWithAffinity
// CHECK-SAME: (%[[TENSOR:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[STORAGE:.+]]: !hal.buffer)
// CHECK-SAME: -> (!stream.resource<*>, index)
util.func public @tensorTransientsWithAffinity(%tensor: tensor<4xf32>, %storage: !hal.buffer) -> tensor<4xf32> {
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[TENSOR]] : !stream.resource<*>{%[[SIZE]]} -> to(#hal.device.promise<@dev>) !stream.resource<*>{%[[SIZE]]}
  // CHECK: %[[BARRIER_RESULT:.+]], %[[BARRIER_TP:.+]] = stream.timepoint.barrier on(#hal.device.promise<@dev>) %[[TRANSFER]] : !stream.resource<*>{%[[SIZE]]} => !stream.timepoint
  // CHECK: %[[BUFFER_LENGTH:.+]] = hal.buffer.length<%[[STORAGE]] : !hal.buffer> : index
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import on(#hal.device.promise<@dev>) %[[STORAGE]] : !hal.buffer -> tensor<?xi8>{%[[BUFFER_LENGTH]]} in !stream.resource<transient>{%[[BUFFER_LENGTH]]}
  // CHECK: %[[TRANSIENTS_RESULT:.+]], %[[TRANSIENTS_TP:.+]] = stream.resource.transients on(#hal.device.promise<@dev>) await(%[[BARRIER_TP]]) => %[[BARRIER_RESULT]] : !stream.resource<*>{%[[SIZE]]} from %[[IMPORTED]] : !stream.resource<transient>{%[[BUFFER_LENGTH]]} => !stream.timepoint
  // CHECK: %[[RESULT:.+]] = stream.timepoint.await %[[TRANSIENTS_TP]] => %[[TRANSIENTS_RESULT]] : !stream.resource<*>{%[[SIZE]]}
  %0 = hal.tensor.transients on(#hal.device.promise<@dev>) %tensor : tensor<4xf32> from %storage : !hal.buffer
  // CHECK: util.return %[[RESULT]], %[[SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @tensorTransientsBufferView
// CHECK-SAME: (%[[TENSOR:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[STORAGE:.+]]: !hal.buffer_view)
// CHECK-SAME: -> (!stream.resource<*>, index)
util.func public @tensorTransientsBufferView(%tensor: tensor<4xf32>, %storage: !hal.buffer_view) -> tensor<4xf32> {
  // CHECK: %[[BARRIER_RESULT:.+]], %[[BARRIER_TP:.+]] = stream.timepoint.barrier %[[TENSOR]] : !stream.resource<*>{%[[SIZE]]} => !stream.timepoint
  // CHECK: %[[BUFFER:.+]] = hal.buffer_view.buffer<%[[STORAGE]] : !hal.buffer_view> : !hal.buffer
  // CHECK: %[[BUFFER_LENGTH:.+]] = hal.buffer.length<%[[BUFFER]] : !hal.buffer> : index
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import %[[BUFFER]] : !hal.buffer -> tensor<?xi8>{%[[BUFFER_LENGTH]]} in !stream.resource<transient>{%[[BUFFER_LENGTH]]}
  // CHECK: %[[TRANSIENTS_RESULT:.+]], %[[TRANSIENTS_TP:.+]] = stream.resource.transients await(%[[BARRIER_TP]]) => %[[BARRIER_RESULT]] : !stream.resource<*>{%[[SIZE]]} from %[[IMPORTED]] : !stream.resource<transient>{%[[BUFFER_LENGTH]]} => !stream.timepoint
  // CHECK: %[[RESULT:.+]] = stream.timepoint.await %[[TRANSIENTS_TP]] => %[[TRANSIENTS_RESULT]] : !stream.resource<*>{%[[SIZE]]}
  %0 = hal.tensor.transients %tensor : tensor<4xf32> from %storage : !hal.buffer_view
  // CHECK: util.return %[[RESULT]], %[[SIZE]] : !stream.resource<*>, index
  util.return %0 : tensor<4xf32>
}
