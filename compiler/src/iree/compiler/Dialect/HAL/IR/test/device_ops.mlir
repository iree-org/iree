// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @device_allocator
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
func.func @device_allocator(%device: !hal.device) -> !hal.allocator {
  // CHECK: %allocator = hal.device.allocator<%[[DEVICE]] : !hal.device> : !hal.allocator
  %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
  return %allocator : !hal.allocator
}

// -----

// CHECK-LABEL: @device_query
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
func.func @device_query(%device : !hal.device) -> (i1, i32) {
  // CHECK: = hal.device.query<%[[DEVICE]] : !hal.device> key("sys" :: "foo") : i1, i32
  %ok, %value = hal.device.query<%device : !hal.device> key("sys" :: "foo") : i1, i32
  return %ok, %value : i1, i32
}

// -----

// CHECK-LABEL: @device_queue_alloca
func.func @device_queue_alloca(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence, %[[SIGNAL_FENCE:.+]]: !hal.fence,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[SIZE:.+]]: index)
    %size: index) -> !hal.buffer {
  %c100_i64 = arith.constant 100 : i64
  // CHECK: = hal.device.queue.alloca<%[[DEVICE]] : !hal.device>
  %buffer = hal.device.queue.alloca<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
      // CHECK-SAME: wait(%[[WAIT_FENCE]]) signal(%[[SIGNAL_FENCE]])
      wait(%wait_fence) signal(%signal_fence)
      // CHECK-SAME: pool(%c100_i64)
      pool(%c100_i64)
      // CHECK-SAME: type({{.+}}) usage({{.+}})
      type(DeviceLocal) usage(Transfer)
      // CHECK-SAME: : !hal.buffer{%[[SIZE]]}
      : !hal.buffer{%size}
  return %buffer : !hal.buffer
}

// -----

// CHECK-LABEL: @device_queue_dealloca
func.func @device_queue_dealloca(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence, %[[SIGNAL_FENCE:.+]]: !hal.fence,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[BUFFER:.+]]: !hal.buffer)
    %buffer: !hal.buffer) {
  // CHECK: hal.device.queue.dealloca<%[[DEVICE]] : !hal.device>
  hal.device.queue.dealloca<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
      // CHECK-SAME: wait(%[[WAIT_FENCE]]) signal(%[[SIGNAL_FENCE]])
      wait(%wait_fence) signal(%signal_fence)
      // CHECK-SAME: buffer(%[[BUFFER]] : !hal.buffer)
      buffer(%buffer : !hal.buffer)
  return
}

// -----

// CHECK-LABEL: @device_queue_read
func.func @device_queue_read(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence, %[[SIGNAL_FENCE:.+]]: !hal.fence,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[SOURCE_FILE:.+]]: !hal.file,
    %source_file: !hal.file,
    // CHECK-SAME:  %[[TARGET_BUFFER:.+]]: !hal.buffer)
    %target_buffer: !hal.buffer) {
  // CHECK-DAG: %[[SOURCE_OFFSET:.+]] = arith.constant 100
  %source_offset = arith.constant 100 : i64
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = arith.constant 200
  %target_offset = arith.constant 200 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 300
  %length = arith.constant 300 : index
  // CHECK: hal.device.queue.read<%[[DEVICE]] : !hal.device>
  hal.device.queue.read<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
      // CHECK-SAME: wait(%[[WAIT_FENCE]]) signal(%[[SIGNAL_FENCE]])
      wait(%wait_fence) signal(%signal_fence)
      // CHECK-SAME: source(%[[SOURCE_FILE]] : !hal.file)[%[[SOURCE_OFFSET]]]
      source(%source_file : !hal.file)[%source_offset]
      // CHECK-SAME: target(%[[TARGET_BUFFER]] : !hal.buffer)[%[[TARGET_OFFSET]]]
      target(%target_buffer : !hal.buffer)[%target_offset]
      // CHECK-SAME: length(%[[LENGTH]])
      length(%length)
      // CHECK-SAME: flags(0)
      flags(0)
  return
}

// -----

// CHECK-LABEL: @device_queue_write
func.func @device_queue_write(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence, %[[SIGNAL_FENCE:.+]]: !hal.fence,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[SOURCE_BUFFER:.+]]: !hal.buffer,
    %source_buffer: !hal.buffer,
    // CHECK-SAME:  %[[TARGET_FILE:.+]]: !hal.file)
    %target_file: !hal.file) {
  // CHECK-DAG: %[[SOURCE_OFFSET:.+]] = arith.constant 100
  %source_offset = arith.constant 100 : index
  // CHECK-DAG: %[[TARGET_OFFSET:.+]] = arith.constant 200
  %target_offset = arith.constant 200 : i64
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 300
  %length = arith.constant 300 : index
  // CHECK: hal.device.queue.write<%[[DEVICE]] : !hal.device>
  hal.device.queue.write<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
      // CHECK-SAME: wait(%[[WAIT_FENCE]]) signal(%[[SIGNAL_FENCE]])
      wait(%wait_fence) signal(%signal_fence)
      // CHECK-SAME: source(%[[SOURCE_BUFFER]] : !hal.buffer)[%[[SOURCE_OFFSET]]]
      source(%source_buffer : !hal.buffer)[%source_offset]
      // CHECK-SAME: target(%[[TARGET_FILE]] : !hal.file)[%[[TARGET_OFFSET]]]
      target(%target_file : !hal.file)[%target_offset]
      // CHECK-SAME: length(%[[LENGTH]])
      length(%length)
      // CHECK-SAME: flags(0)
      flags(0)
  return
}

// -----

// CHECK-LABEL: @device_queue_execute
func.func @device_queue_execute(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence, %[[SIGNAL_FENCE:.+]]: !hal.fence,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME:  %[[CMD0:.+]]: !hal.command_buffer, %[[CMD1:.+]]: !hal.command_buffer)
    %cmd0: !hal.command_buffer, %cmd1: !hal.command_buffer) {
  // CHECK: hal.device.queue.execute<%[[DEVICE]] : !hal.device>
  hal.device.queue.execute<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
      // CHECK-SAME: wait(%[[WAIT_FENCE]]) signal(%[[SIGNAL_FENCE]])
      wait(%wait_fence) signal(%signal_fence)
      // CHECK-SAME: commands([%[[CMD0]], %[[CMD1]]])
      commands([%cmd0, %cmd1])
  return
}

// -----

// CHECK-LABEL: @device_queue_flush
func.func @device_queue_flush(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64)
    %device: !hal.device, %affinity: i64) {
  // CHECK: hal.device.queue.flush<%[[DEVICE]] : !hal.device>
  hal.device.queue.flush<%device : !hal.device>
      // CHECK-SAME: affinity(%[[AFFINITY]])
      affinity(%affinity)
  return
}
