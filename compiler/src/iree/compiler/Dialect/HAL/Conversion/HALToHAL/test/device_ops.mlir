// RUN: iree-opt --split-input-file --allow-unregistered-dialect --iree-hal-conversion %s | FileCheck %s

util.global private @device : !hal.device

// CHECK-LABEL: @deviceResolveDevice
util.func public @deviceResolveDevice() -> !hal.device {
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  %device = hal.device.resolve on(#hal.device.affinity<@device>) : !hal.device
  // CHECK: util.return %[[DEVICE]]
  util.return %device : !hal.device
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @deviceResolveDeviceQueueAffinityAny
util.func public @deviceResolveDeviceQueueAffinityAny() -> (!hal.device, i64) {
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[QUEUE_AFFINITY:.+]] = arith.constant -1 : i64
  %device, %queue_affinity_any = hal.device.resolve on(#hal.device.affinity<@device>) : !hal.device, i64
  // CHECK: util.return %[[DEVICE]], %[[QUEUE_AFFINITY]]
  util.return %device, %queue_affinity_any : !hal.device, i64
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @deviceResolveDeviceQueueAffinity45
util.func public @deviceResolveDeviceQueueAffinity45() -> (!hal.device, i64) {
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[QUEUE_AFFINITY:.+]] = arith.constant 48 : i64
  %device, %queue_affinity_45 = hal.device.resolve on(#hal.device.affinity<@device, [4, 5]>) : !hal.device, i64
  // CHECK: util.return %[[DEVICE]], %[[QUEUE_AFFINITY]]
  util.return %device, %queue_affinity_45 : !hal.device, i64
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @deviceResolveAllocator
util.func public @deviceResolveAllocator() -> !hal.allocator {
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[ALLOCATOR:.+]] = hal.device.allocator<%[[DEVICE]] : !hal.device> : !hal.allocator
  %allocator = hal.device.resolve on(#hal.device.affinity<@device>) : !hal.allocator
  // CHECK: util.return %[[ALLOCATOR]]
  util.return %allocator : !hal.allocator
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @deviceResolveAllocatorQueueAffinity45
util.func public @deviceResolveAllocatorQueueAffinity45() -> (!hal.allocator, i64) {
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[ALLOCATOR:.+]] = hal.device.allocator<%[[DEVICE]] : !hal.device> : !hal.allocator
  // CHECK-DAG: %[[QUEUE_AFFINITY:.+]] = arith.constant 48 : i64
  %allocator, %queue_affinity_45 = hal.device.resolve on(#hal.device.affinity<@device, [4, 5]>) : !hal.allocator, i64
  // CHECK: util.return %[[ALLOCATOR]], %[[QUEUE_AFFINITY]]
  util.return %allocator, %queue_affinity_45 : !hal.allocator, i64
}
