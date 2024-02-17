// RUN: iree-opt --split-input-file --canonicalize -cse %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @FoldBufferLengthOp
// CHECK-SAME: (%[[LENGTH:.+]]: index)
util.func public @FoldBufferLengthOp(%length: index) -> index {
  %c64 = arith.constant 64 : index
  %buffer, %storage = hal_inline.buffer.allocate alignment(%c64) : !hal.buffer{%length} in !util.buffer
  // CHECK-NOT: hal_inline.buffer.length
  %queried_length = hal_inline.buffer.length<%buffer : !hal.buffer> : index
  // CHECK: return %[[LENGTH]]
  util.return %queried_length : index
}

// -----

// CHECK-LABEL: @FoldBufferStorageOp
util.func public @FoldBufferStorageOp(%length: index) -> !util.buffer {
  %c64 = arith.constant 64 : index
  // CHECK: %[[BUFFER:.+]], %[[STORAGE:.+]] = hal_inline.buffer.allocate
  %buffer, %storage = hal_inline.buffer.allocate alignment(%c64) : !hal.buffer{%length} in !util.buffer
  // CHECK-NOT: hal_inline.buffer.storage
  %queried_storage = hal_inline.buffer.storage<%buffer : !hal.buffer> : !util.buffer
  // CHECK: return %[[STORAGE]]
  util.return %queried_storage : !util.buffer
}

// -----

// CHECK-LABEL: @FoldBufferViewCreateSubspan
// CHECK-SAME: (%[[BASE_BUFFER:.+]]: !hal.buffer, %[[SUBSPAN_OFFSET:.+]]: index, %[[SUBSPAN_LENGTH:.+]]: index)
util.func public @FoldBufferViewCreateSubspan(%base_buffer: !hal.buffer, %subspan_offset: index, %subspan_length: index) -> !hal.buffer_view {
  %subspan = hal_inline.buffer.subspan<%base_buffer : !hal.buffer>[%subspan_offset, %subspan_length] : !hal.buffer
  // CHECK-DAG: %[[VIEW_OFFSET:.+]] = arith.constant 512
  %view_offset = arith.constant 512 : index
  // CHECK-DAG: %[[VIEW_LENGTH:.+]] = arith.constant 1024
  %view_length = arith.constant 1024 : index
  // CHECK-DAG: %[[FOLDED_OFFSET:.+]] = arith.addi %[[SUBSPAN_OFFSET]], %[[VIEW_OFFSET]]
  // CHECK: = hal_inline.buffer_view.create
  // CHECK-SAME: buffer(%[[BASE_BUFFER]] : !hal.buffer)[%[[FOLDED_OFFSET]], %[[VIEW_LENGTH]]]
  %dim0 = arith.constant 128 : index
  %type = arith.constant 32 : i32
  %encoding = arith.constant 1 : i32
  %view = hal_inline.buffer_view.create buffer(%subspan : !hal.buffer)[%view_offset, %view_length]
                                 shape([%dim0])
                                 type(%type)
                                 encoding(%encoding) : !hal.buffer_view
  util.return %view : !hal.buffer_view
}

// -----

// CHECK-LABEL: @SkipBufferViewBufferOp
// CHECK-SAME: (%[[BUFFER:.+]]: !hal.buffer)
util.func public @SkipBufferViewBufferOp(%buffer: !hal.buffer) -> !hal.buffer {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : index
  %c11 = arith.constant 11 : index
  %c32 = arith.constant 32 : i32
  %c64 = arith.constant 64 : index
  %view = hal_inline.buffer_view.create buffer(%buffer : !hal.buffer)[%c0, %c64]
                                        shape([%c10, %c11])
                                        type(%c32)
                                        encoding(%c1) : !hal.buffer_view
  %view_buffer = hal_inline.buffer_view.buffer<%view : !hal.buffer_view> : !hal.buffer
  // CHECK: return %[[BUFFER]]
  util.return %view_buffer : !hal.buffer
}
