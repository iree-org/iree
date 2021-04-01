// Tests printing and parsing of hal.buffer ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @buffer_allocator
func @buffer_allocator(%arg0: !hal.buffer) -> !hal.allocator {
  // CHECK: %allocator = hal.buffer.allocator<%arg0 : !hal.buffer> : !hal.allocator
  %allocator = hal.buffer.allocator<%arg0 : !hal.buffer> : !hal.allocator
  return %allocator : !hal.allocator
}

// -----

// CHECK-LABEL: @buffer_subspan
func @buffer_subspan(%arg0: !hal.buffer) -> !hal.buffer {
  // CHECK-DAG: %[[OFFSET:.+]] = constant 100
  %offset = constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = constant 200
  %length = constant 200 : index
  // CHECK: %buffer = hal.buffer.subspan<%arg0 : !hal.buffer>[%[[OFFSET]], %[[LENGTH]]] : !hal.buffer
  %buffer = hal.buffer.subspan<%arg0 : !hal.buffer>[%offset, %length] : !hal.buffer
  return %buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @buffer_length
func @buffer_length(%arg0: !hal.buffer) -> index {
  // CHECK: hal.buffer.length<%arg0 : !hal.buffer> : index
  %length = hal.buffer.length<%arg0 : !hal.buffer> : index
  return %length : index
}

// -----

// CHECK-LABEL: @buffer_fill
func @buffer_fill(%arg0: !hal.buffer) {
  // CHECK-DAG: %[[OFFSET:.+]] = constant 100
  %offset = constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = constant 200
  %length = constant 200 : index
  // CHECK-DAG: %[[PATTERN:.+]] = constant 42
  %pattern = constant 42 : i32
  // CHECK: hal.buffer.fill<%arg0 : !hal.buffer>[%[[OFFSET]], %[[LENGTH]]] pattern(%[[PATTERN]] : i32)
  hal.buffer.fill<%arg0 : !hal.buffer>[%offset, %length] pattern(%pattern : i32)
  return
}

// -----

// CHECK-LABEL: @buffer_copy
func @buffer_copy(%arg0: !hal.buffer, %arg1: !hal.buffer) {
  // CHECK-DAG: %[[SRC_OFFSET:.+]] = constant 100
  %src_offset = constant 100 : index
  // CHECK-DAG: %[[DST_OFFSET:.+]] = constant 200
  %dst_offset = constant 200 : index
  // CHECK-DAG: %[[LENGTH:.+]] = constant 300
  %length = constant 300 : index
  // CHECK: hal.buffer.copy source(%arg0 : !hal.buffer)[%[[SRC_OFFSET]]]
  // CHECK-SAME:            target(%arg1 : !hal.buffer)[%[[DST_OFFSET]]]
  // CHECK-SAME:            length(%[[LENGTH]])
  hal.buffer.copy source(%arg0 : !hal.buffer)[%src_offset]
                  target(%arg1 : !hal.buffer)[%dst_offset]
                  length(%length)
  return
}

// -----

// CHECK-LABEL: @buffer_load
func @buffer_load(%arg0: !hal.buffer) -> i32 {
  // CHECK-DAG: %[[SRC_OFFSET:.+]] = constant 100
  %src_offset = constant 100 : index
  // CHECK: %[[VAL:.+]] = hal.buffer.load<%arg0 : !hal.buffer>[%[[SRC_OFFSET]]] : i32
  %1 = hal.buffer.load<%arg0 : !hal.buffer>[%src_offset] : i32
  // CHECK-NEXT: return %[[VAL]]
  return %1 : i32
}

// -----

// CHECK-LABEL: @buffer_store
func @buffer_store(%arg0: !hal.buffer, %arg1: i32) {
  // CHECK-DAG: %[[DST_OFFSET:.+]] = constant 100
  %dst_offset = constant 100 : index
  // CHECK: hal.buffer.store<%arg0 : !hal.buffer>[%[[DST_OFFSET]]] value(%arg1 : i32)
  hal.buffer.store<%arg0 : !hal.buffer>[%dst_offset] value(%arg1 : i32)
  return
}
