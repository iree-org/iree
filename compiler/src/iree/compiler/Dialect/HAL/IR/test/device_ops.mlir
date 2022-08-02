// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @device_allocator
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
func.func @device_allocator(%device: !hal.device) -> !hal.allocator {
  // CHECK: %allocator = hal.device.allocator<%[[DEVICE]] : !hal.device> : !hal.allocator
  %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
  return %allocator : !hal.allocator
}

// -----

// CHECK-LABEL: @device_switch
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
func.func @device_switch(%device: !hal.device) -> i32 {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0
  %c0 = arith.constant 0 : i32
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1
  %c1 = arith.constant 1 : i32
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2
  %c2 = arith.constant 2 : i32
  // CHECK: = hal.device.switch<%[[DEVICE]] : !hal.device> -> i32
  %0 = hal.device.switch<%device : !hal.device> -> i32
    // CHECK-NEXT: #hal.device.match.id<"vulkan-v1.?-*"> {
    #hal.device.match.id<"vulkan-v1.?-*"> {
      // CHECK-NEXT: hal.return %[[C1]] : i32
      hal.return %c1 : i32
      // CHECK-NEXT: },
    },
    // CHECK-NEXT: #hal.match.any<[#hal.device.match.id<"vmvx">, #hal.device.match.id<"vulkan-*">]> {
    #hal.match.any<[#hal.device.match.id<"vmvx">, #hal.device.match.id<"vulkan-*">]> {
      // CHECK-NEXT: hal.return %[[C2]] : i32
      hal.return %c2 : i32
      // CHECK-NEXT: },
    },
    // CHECK-NEXT: #hal.match.always {
    #hal.match.always {
      // CHECK-NEXT: hal.return %[[C0]] : i32
      hal.return %c0 : i32
      // CHECK-NEXT: }
    }
  return %0 : i32
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
    // CHECK-SAME: %[[SIZE:.+]]: index)
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
    // CHECK-SAME: %[[BUFFER:.+]]: !hal.buffer)
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

// CHECK-LABEL: @device_queue_execute
func.func @device_queue_execute(
    // CHECK-SAME: (%[[DEVICE:.+]]: !hal.device, %[[AFFINITY:.+]]: i64,
    %device: !hal.device, %affinity: i64,
    // CHECK-SAME:  %[[WAIT_FENCE:.+]]: !hal.fence, %[[SIGNAL_FENCE:.+]]: !hal.fence,
    %wait_fence: !hal.fence, %signal_fence: !hal.fence,
    // CHECK-SAME: %[[CMD0:.+]]: !hal.command_buffer, %[[CMD1:.+]]: !hal.command_buffer)
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
