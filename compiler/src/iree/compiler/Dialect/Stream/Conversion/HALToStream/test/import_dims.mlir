// RUN: iree-opt --split-input-file --canonicalize --iree-stream-conversion --canonicalize %s | FileCheck %s

// CHECK-LABEL: @importBufferViewWithConsumerDims
// CHECK-SAME: (%[[VIEW:.+]]: !hal.buffer_view, %[[STORAGE:.+]]: !hal.buffer)
// CHECK-SAME: -> (!stream.resource<*>, index)
util.func public @importBufferViewWithConsumerDims(%view: !hal.buffer_view, %storage: !hal.buffer) -> tensor<?x?xf32> {
  %dim0 = hal.buffer_view.dim<%view : !hal.buffer_view>[0] : index
  %dim1 = hal.buffer_view.dim<%view : !hal.buffer_view>[1] : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
  // CHECK-DAG: %[[IMPORT_SIZE:.+]] = stream.tensor.sizeof tensor<?x?xf32>{%[[C4]], %[[C8]]} : index
  // CHECK: %[[RESOURCE:.+]] = stream.tensor.import %[[VIEW]]
  // CHECK-SAME: tensor<?x?xf32>{%[[C4]], %[[C8]]} in !stream.resource<external>{%[[IMPORT_SIZE]]}
  // CHECK: stream.async.clone %[[RESOURCE]]
  %imported = hal.tensor.import %view : !hal.buffer_view -> tensor<?x?xf32>{%dim0, %dim1}
  %aliased = hal.tensor.alias %imported : tensor<?x?xf32>{%c4, %c8} to %storage : !hal.buffer
  // CHECK: util.return
  util.return %aliased : tensor<?x?xf32>
}
