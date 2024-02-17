// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @buffer_subspan
util.func public @buffer_subspan(%arg0: !hal.buffer) -> !hal.buffer {
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 100
  %offset = arith.constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 200
  %length = arith.constant 200 : index
  // CHECK: %buffer = hal.buffer.subspan<%arg0 : !hal.buffer>[%[[OFFSET]], %[[LENGTH]]] : !hal.buffer
  %buffer = hal.buffer.subspan<%arg0 : !hal.buffer>[%offset, %length] : !hal.buffer
  util.return %buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @buffer_length
util.func public @buffer_length(%arg0: !hal.buffer) -> index {
  // CHECK: hal.buffer.length<%arg0 : !hal.buffer> : index
  %length = hal.buffer.length<%arg0 : !hal.buffer> : index
  util.return %length : index
}

// -----

// CHECK-LABEL: @buffer_load
util.func public @buffer_load(%arg0: !hal.buffer) -> i32 {
  // CHECK-DAG: %[[SRC_OFFSET:.+]] = arith.constant 100
  %src_offset = arith.constant 100 : index
  // CHECK: %[[VAL:.+]] = hal.buffer.load<%arg0 : !hal.buffer>[%[[SRC_OFFSET]]] : i32
  %1 = hal.buffer.load<%arg0 : !hal.buffer>[%src_offset] : i32
  // CHECK-NEXT: util.return %[[VAL]]
  util.return %1 : i32
}

// -----

// CHECK-LABEL: @buffer_store
util.func public @buffer_store(%arg0: !hal.buffer, %arg1: i32) {
  // CHECK-DAG: %[[DST_OFFSET:.+]] = arith.constant 100
  %dst_offset = arith.constant 100 : index
  // CHECK: hal.buffer.store<%arg0 : !hal.buffer>[%[[DST_OFFSET]]] value(%arg1 : i32)
  hal.buffer.store<%arg0 : !hal.buffer>[%dst_offset] value(%arg1 : i32)
  util.return
}
