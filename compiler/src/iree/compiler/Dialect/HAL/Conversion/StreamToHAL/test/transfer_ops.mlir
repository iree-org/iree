// RUN: iree-opt --split-input-file --iree-hal-conversion %s | FileCheck %s

// CHECK-LABEL: @tensorImportBuffer
util.func public @tensorImportBuffer(%arg0: !hal.buffer, %arg1: index) -> !stream.resource<external> {
  %c20 = arith.constant 20 : index
  // CHECK-DAG: %[[ALLOCATOR:.+]] = hal.device.allocator
  // CHECK: hal.buffer.assert<%arg0 : !hal.buffer>
  // CHECK-SAME: message("tensor")
  // CHECK-SAME: allocator(%[[ALLOCATOR]] : !hal.allocator)
  // CHECK-SAME: minimum_length(%c20)
  // CHECK-SAME: type(DeviceVisible)
  // CHECK-SAME: usage("Transfer{{.+}}Dispatch{{.+}}")
  %0 = stream.tensor.import %arg0 : !hal.buffer -> tensor<?x5xf32>{%arg1} in !stream.resource<external>{%c20}
  // CHECK: util.return %arg0
  util.return %0 : !stream.resource<external>
}

// -----

// NOTE: buffer view metadata assertions via hal.buffer_view.assert are added
// when lowering into the stream dialect; here we only care about the storage
// buffer itself.

// CHECK-LABEL: @tensorImportBufferView
util.func public @tensorImportBufferView(%arg0: !hal.buffer_view, %arg1: index) -> !stream.resource<external> {
  %c20 = arith.constant 20 : index
  // CHECK-DAG: %[[BUFFER:.+]] = hal.buffer_view.buffer<%arg0 : !hal.buffer_view> : !hal.buffer
  // CHECK-DAG: %[[ALLOCATOR:.+]] = hal.device.allocator
  // CHECK: hal.buffer.assert<%[[BUFFER]] : !hal.buffer>
  // CHECK-SAME: message("tensor")
  // CHECK-SAME: allocator(%[[ALLOCATOR]] : !hal.allocator)
  // CHECK-SAME: minimum_length(%c20)
  // CHECK-SAME: type(DeviceVisible)
  // CHECK-SAME: usage("Transfer{{.+}}Dispatch{{.+}}")
  %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<?x5xf32>{%arg1} in !stream.resource<external>{%c20}
  // CHECK: util.return %[[BUFFER]]
  util.return %0 : !stream.resource<external>
}

// -----

// CHECK-LABEL: @tensorExportBuffer
util.func public @tensorExportBuffer(%arg0: !stream.resource<external>, %arg1: index) -> !hal.buffer {
  %c200 = arith.constant 200 : index
  %0 = stream.tensor.export %arg0 : tensor<?x1x10xf32>{%arg1} in !stream.resource<external>{%c200} -> !hal.buffer
  // CHECK: util.return %arg0 : !hal.buffer
  util.return %0 : !hal.buffer
}

// -----

// CHECK-LABEL: @tensorExportBufferView
util.func public @tensorExportBufferView(%arg0: !stream.resource<external>, %arg1: index) -> !hal.buffer_view {
  %c200 = arith.constant 200 : index
  // CHECK-DAG: %[[ELEMENT_TYPE:.+]] = hal.element_type<f32> : i32
  // CHECK-DAG: %[[ENCODING_TYPE:.+]] = hal.encoding_type<dense_row_major> : i32
  // CHECK: %[[VIEW:.+]] = hal.buffer_view.create
  // CHECK-SAME: buffer(%arg0 : !hal.buffer)
  // CHECK-SAME: shape([%arg1, %c1, %c10])
  // CHECK-SAME: type(%[[ELEMENT_TYPE]])
  // CHECK-SAME: encoding(%[[ENCODING_TYPE]])
  // CHECK-SAME: : !hal.buffer_view
  %0 = stream.tensor.export %arg0 : tensor<?x1x10xf32>{%arg1} in !stream.resource<external>{%c200} -> !hal.buffer_view
  // CHECK: util.return %[[VIEW]]
  util.return %0 : !hal.buffer_view
}
