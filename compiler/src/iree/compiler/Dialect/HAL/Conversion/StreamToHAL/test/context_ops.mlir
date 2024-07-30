// RUN: iree-opt --split-input-file --allow-unregistered-dialect --iree-hal-conversion %s | FileCheck %s

// NOTE: the hal.device.resolve lowering in HAL-to-HAL does most of the work.

util.global private @device : !hal.device

// CHECK-LABEL: @contextResolveDefaultDevice
util.func public @contextResolveDefaultDevice() -> !hal.device attributes {
  stream.affinity = #hal.device.affinity<@device>
} {
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  %device = stream.context.resolve : !hal.device
  // CHECK: util.return %[[DEVICE]]
  util.return %device : !hal.device
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @contextResolveAllocatorQueueAffinity45
util.func public @contextResolveAllocatorQueueAffinity45() -> (!hal.device, !hal.allocator, i64) {
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK-DAG: %[[ALLOCATOR:.+]] = hal.device.allocator<%[[DEVICE]] : !hal.device> : !hal.allocator
  // CHECK-DAG: %[[QUEUE_AFFINITY:.+]] = arith.constant 48 : i64
  %device, %allocator, %queue_affinity_45 = stream.context.resolve on(#hal.device.affinity<@device, [4, 5]>) : !hal.device, !hal.allocator, i64
  // CHECK: util.return %[[DEVICE]], %[[ALLOCATOR]], %[[QUEUE_AFFINITY]]
  util.return %device, %allocator, %queue_affinity_45 : !hal.device, !hal.allocator, i64
}
