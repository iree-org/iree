// RUN: iree-opt --split-input-file --canonicalize -cse %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @FoldBufferViewCreateSubspan
// CHECK-SAME: (%[[BASE_BUFFER:.+]]: !hal.buffer, %[[SUBSPAN_OFFSET:.+]]: index, %[[SUBSPAN_LENGTH:.+]]: index)
util.func public @FoldBufferViewCreateSubspan(%base_buffer: !hal.buffer, %subspan_offset: index, %subspan_length: index) -> !hal.buffer_view {
  %subspan = hal.buffer.subspan<%base_buffer : !hal.buffer>[%subspan_offset, %subspan_length] : !hal.buffer
  // CHECK-DAG: %[[VIEW_OFFSET:.+]] = arith.constant 512
  %view_offset = arith.constant 512 : index
  // CHECK-DAG: %[[VIEW_LENGTH:.+]] = arith.constant 1024
  %view_length = arith.constant 1024 : index
  // CHECK-DAG: %[[FOLDED_OFFSET:.+]] = arith.addi %[[SUBSPAN_OFFSET]], %[[VIEW_OFFSET]]
  // CHECK: = hal.buffer_view.create
  // CHECK-SAME: buffer(%[[BASE_BUFFER]] : !hal.buffer)[%[[FOLDED_OFFSET]], %[[VIEW_LENGTH]]]
  %dim0 = arith.constant 128 : index
  %type = arith.constant 32 : i32
  %encoding = arith.constant 1 : i32
  %view = hal.buffer_view.create buffer(%subspan : !hal.buffer)[%view_offset, %view_length]
                                 shape([%dim0])
                                 type(%type)
                                 encoding(%encoding) : !hal.buffer_view
  util.return %view : !hal.buffer_view
}
