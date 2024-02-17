// RUN: iree-opt --split-input-file --iree-hal-inline-conversion %s | FileCheck %s

// CHECK-LABEL: @tensorImportBuffer
// CHECK-SAME: (%[[BUFFER:.+]]: !hal.buffer, %[[RESOURCE_SIZE:.+]]: index, %[[DIM:.+]]: index) -> !hal.buffer
util.func public @tensorImportBuffer(%buffer: !hal.buffer, %resource_size: index, %dim: index) -> !stream.resource<external> {
  %0 = stream.tensor.import %buffer : !hal.buffer -> tensor<?x5xf32>{%dim} in !stream.resource<external>{%resource_size}
  // CHECK: return %[[BUFFER]]
  util.return %0 : !stream.resource<external>
}

// -----

// NOTE: buffer view metadata assertions via hal.buffer_view.assert are added
// when lowering into the stream dialect; here we only care about the storage
// buffer itself.

// CHECK-LABEL: @tensorImportBufferView
// CHECK-SAME: (%[[BUFFER_VIEW:.+]]: !hal.buffer_view, %[[RESOURCE_SIZE:.+]]: index, %[[DIM:.+]]: index) -> !hal.buffer
util.func public @tensorImportBufferView(%buffer_view: !hal.buffer_view, %resource_size: index, %dim: index) -> !stream.resource<external> {
  // CHECK: %[[BUFFER:.+]] = hal_inline.buffer_view.buffer<%[[BUFFER_VIEW]] : !hal.buffer_view> : !hal.buffer
  %0 = stream.tensor.import %buffer_view : !hal.buffer_view -> tensor<?x5xf32>{%dim} in !stream.resource<external>{%resource_size}
  // CHECK: return %[[BUFFER]]
  util.return %0 : !stream.resource<external>
}

// -----

// CHECK-LABEL: @tensorExportBuffer
// CHECK-SAME: (%[[BUFFER:.+]]: !hal.buffer, %[[RESOURCE_SIZE:.+]]: index, %[[DIM:.+]]: index) -> !hal.buffer
util.func public @tensorExportBuffer(%resource: !stream.resource<external>, %resource_size: index, %dim: index) -> !hal.buffer {
  %0 = stream.tensor.export %resource : tensor<?x1x10xf32>{%dim} in !stream.resource<external>{%resource_size} -> !hal.buffer
  // CHECK: return %[[BUFFER]]
  util.return %0 : !hal.buffer
}

// -----

// CHECK-LABEL: @tensorExportBufferView
// CHECK-SAME: (%[[BUFFER:.+]]: !hal.buffer, %[[RESOURCE_SIZE:.+]]: index, %[[DIM:.+]]: index) -> !hal.buffer
util.func public @tensorExportBufferView(%resource: !stream.resource<external>, %resource_size: index, %dim: index) -> !hal.buffer_view {
  // CHECK: %[[BUFFER_VIEW:.+]] = hal_inline.buffer_view.create
  // CHECK-SAME: buffer(%[[BUFFER]] : !hal.buffer)
  // CHECK-SAME: shape([%[[DIM]], %c1, %c10])
  // CHECK-SAME: type(%c553648160_i32)
  // CHECK-SAME: encoding(%c1_i32)
  %0 = stream.tensor.export %resource : tensor<?x1x10xf32>{%dim} in !stream.resource<external>{%resource_size} -> !hal.buffer_view
  // CHECK: return %[[BUFFER_VIEW]]
  util.return %0 : !hal.buffer_view
}
