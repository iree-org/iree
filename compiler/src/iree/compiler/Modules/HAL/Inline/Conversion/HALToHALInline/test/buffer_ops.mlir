// RUN: iree-opt --split-input-file --iree-hal-inline-conversion %s | FileCheck %s

// CHECK-LABEL: @buffer_subspan
// CHECK-SAME: (%[[BUFFER:.+]]: !hal.buffer)
util.func public @buffer_subspan(%buffer: !hal.buffer) -> !hal.buffer {
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 100
  %offset = arith.constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 200
  %length = arith.constant 200 : index
  // CHECK: %[[SUBSPAN:.+]] = hal_inline.buffer.subspan<%[[BUFFER]] : !hal.buffer>[%[[OFFSET]], %[[LENGTH]]] : !hal.buffer
  %subspan = hal.buffer.subspan<%buffer : !hal.buffer>[%offset, %length] : !hal.buffer
  // CHECK: return %[[SUBSPAN]]
  util.return %subspan : !hal.buffer
}

// -----

// CHECK-LABEL: @buffer_length
// CHECK-SAME: (%[[BUFFER:.+]]: !hal.buffer)
util.func public @buffer_length(%buffer: !hal.buffer) -> index {
  // CHECK: hal_inline.buffer.length<%[[BUFFER]] : !hal.buffer> : index
  %length = hal.buffer.length<%buffer : !hal.buffer> : index
  util.return %length : index
}

// -----

// CHECK-LABEL: @buffer_load
// CHECK-SAME: (%[[BUFFER:.+]]: !hal.buffer)
util.func public @buffer_load(%buffer: !hal.buffer) -> i32 {
  // CHECK-DAG: %[[REL_OFFSET:.+]] = arith.constant 100
  %rel_offset = arith.constant 100 : index
  // CHECK-DAG: %[[STORAGE:.+]] = hal_inline.buffer.storage<%[[BUFFER:.+]] : !hal.buffer> : !util.buffer
  // CHECK-DAG: %[[LENGTH:.+]] = hal_inline.buffer.length<%[[BUFFER]] : !hal.buffer> : index
  // CHECK: %[[VALUE:.+]] = util.buffer.load %[[STORAGE]][%[[REL_OFFSET]] for {{.+}}] : !util.buffer{%[[LENGTH]]} -> i32
  %value = hal.buffer.load<%buffer : !hal.buffer>[%rel_offset] : i32
  // CHECK-NEXT: return %[[VALUE]]
  util.return %value : i32
}

// -----

// CHECK-LABEL: @buffer_store
// CHECK-SAME: (%[[BUFFER:.+]]: !hal.buffer, %[[VALUE:.+]]: i32)
util.func public @buffer_store(%buffer: !hal.buffer, %value: i32) {
  // CHECK-DAG: %[[REL_OFFSET:.+]] = arith.constant 100
  %rel_offset = arith.constant 100 : index
  // CHECK-DAG: %[[STORAGE:.+]] = hal_inline.buffer.storage<%[[BUFFER:.+]] : !hal.buffer> : !util.buffer
  // CHECK-DAG: %[[LENGTH:.+]] = hal_inline.buffer.length<%[[BUFFER]] : !hal.buffer> : index
  // CHECK: util.buffer.store %[[VALUE]], %[[STORAGE]][%[[REL_OFFSET]] for {{.+}}] : i32 -> !util.buffer{%[[LENGTH]]}
  hal.buffer.store<%buffer : !hal.buffer>[%rel_offset] value(%value : i32)
  util.return
}
