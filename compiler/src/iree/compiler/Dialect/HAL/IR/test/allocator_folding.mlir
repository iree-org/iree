// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @FoldAllocatorSelectAttr1
util.func public @FoldAllocatorSelectAttr1() -> (!hal.device, i64) {
  // CHECK-NOT: hal.allocator.select.attr
  // CHECK: %[[DEVICE:.+]], %[[QUEUE_AFFINITY:.+]] = hal.device.resolve
  // CHECK-SAME: on(#hal.device.affinity<@device_a, [0]>) : !hal.device, i64
  %memory_type = hal.memory_type<"HostLocal"> : i32
  %buffer_usage = hal.buffer_usage<"Transfer"> : i32
  %device, %queue_affinity = hal.allocator.select.attr
      from(#hal.device.optimal<[#hal.device.affinity<@device_a, [0]>]>)
      type(%memory_type) usage(%buffer_usage) : !hal.device, i64
  // CHECK: util.return %[[DEVICE]], %[[QUEUE_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}

// -----

// CHECK-LABEL: @FoldAllocatorSelectAttrSameDevice
util.func public @FoldAllocatorSelectAttrSameDevice() -> (!hal.device, i64) {
  // CHECK: %[[DEVICE:.+]] = hal.device.resolve
  // CHECK-SAME: on(#hal.device.affinity<@device_a>) : !hal.device
  // CHECK: %[[UNUSED:.+]], %[[QUEUE_AFFINITY:.+]] = hal.allocator.select.attr
  %memory_type = hal.memory_type<"HostLocal"> : i32
  %buffer_usage = hal.buffer_usage<"Transfer"> : i32
  %device, %queue_affinity = hal.allocator.select.attr
      from(#hal.device.optimal<[#hal.device.affinity<@device_a, [0]>, #hal.device.affinity<@device_a, [1]>]>)
      type(%memory_type) usage(%buffer_usage) : !hal.device, i64
  // CHECK: util.return %[[DEVICE]], %[[QUEUE_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}

// -----

// CHECK-LABEL: @FoldAllocatorSelectAttrSameQueueAffinity
util.func public @FoldAllocatorSelectAttrSameQueueAffinity() -> (!hal.device, i64) {
  // CHECK: %[[QUEUE_AFFINITY:.+]] = arith.constant 1 : i64
  // CHECK: %[[DEVICE:.+]], %[[UNUSED:.+]] = hal.allocator.select.attr
  %memory_type = hal.memory_type<"HostLocal"> : i32
  %buffer_usage = hal.buffer_usage<"Transfer"> : i32
  %device, %queue_affinity = hal.allocator.select.attr
      from(#hal.device.optimal<[#hal.device.affinity<@device_a, [0]>, #hal.device.affinity<@device_b, [0]>]>)
      type(%memory_type) usage(%buffer_usage) : !hal.device, i64
  // CHECK: util.return %[[DEVICE]], %[[QUEUE_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}

// -----

// CHECK-LABEL: @FoldAllocatorSelect1
// CHECK-SAME: (%[[SAME_DEVICE:.+]]: !hal.device, %[[SAME_AFFINITY:.+]]: i64)
util.func public @FoldAllocatorSelect1(%same_device: !hal.device, %same_affinity: i64) -> (!hal.device, i64) {
  %type = arith.constant 2 : i32
  %usage = arith.constant 3 : i32
  // CHECK-NOT: hal.allocator.select
  %device, %queue_affinity = hal.allocator.select
      from([
        (%same_device, %same_affinity : !hal.device, i64)
      ])
      type(%type) usage(%usage) : !hal.device, i64
  // CHECK: util.return %[[SAME_DEVICE]], %[[SAME_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}

// -----

// CHECK-LABEL: @FoldAllocatorSelectSameDevice
// CHECK-SAME: (%[[SAME_DEVICE:.+]]: !hal.device, %[[AFFINITY_A:.+]]: i64, %[[AFFINITY_B:.+]]: i64)
util.func public @FoldAllocatorSelectSameDevice(%same_device: !hal.device, %affinity_a: i64, %affinity_b: i64) -> (!hal.device, i64) {
  // CHECK: %[[UNUSED:.+]], %[[QUEUE_AFFINITY:.+]] = hal.allocator.select
  %type = arith.constant 2 : i32
  %usage = arith.constant 3 : i32
  %device, %queue_affinity = hal.allocator.select
      from([
        (%same_device, %affinity_a : !hal.device, i64),
        (%same_device, %affinity_b : !hal.device, i64)
      ])
      type(%type) usage(%usage) : !hal.device, i64
  // CHECK: util.return %[[SAME_DEVICE]], %[[QUEUE_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}

// -----

// CHECK-LABEL: @FoldAllocatorSelectSameQueueAffinity
// CHECK-SAME: (%[[DEVICE_A:.+]]: !hal.device, %[[DEVICE_B:.+]]: !hal.device, %[[SAME_AFFINITY:.+]]: i64)
util.func public @FoldAllocatorSelectSameQueueAffinity(%device_a: !hal.device, %device_b: !hal.device, %same_affinity: i64) -> (!hal.device, i64) {
  // CHECK: %[[DEVICE:.+]], %[[UNUSED:.+]] = hal.allocator.select
  %type = arith.constant 2 : i32
  %usage = arith.constant 3 : i32
  %device, %queue_affinity = hal.allocator.select
      from([
        (%device_a, %same_affinity : !hal.device, i64),
        (%device_b, %same_affinity : !hal.device, i64)
      ])
      type(%type) usage(%usage) : !hal.device, i64
  // CHECK: util.return %[[DEVICE]], %[[SAME_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}
