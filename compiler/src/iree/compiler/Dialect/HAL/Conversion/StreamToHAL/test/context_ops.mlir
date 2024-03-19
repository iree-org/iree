// RUN: iree-opt --split-input-file --allow-unregistered-dialect --iree-hal-conversion %s | FileCheck %s

// CHECK-LABEL: @contextResolveAllocator
util.func public @contextResolveAllocator() -> !hal.allocator {
  // CHECK: %[[DEVICE:.+]] = hal.devices.get %{{.+}}
  // CHECK: %[[ALLOCATOR:.+]] = hal.device.allocator<%[[DEVICE]] : !hal.device> : !hal.allocator
  %allocator = stream.context.resolve : !hal.allocator
  // CHECK: util.return %[[ALLOCATOR]]
  util.return %allocator : !hal.allocator
}

// -----

// CHECK-LABEL: @contextResolveDevice
util.func public @contextResolveDevice() -> !hal.device {
  // CHECK: %[[DEVICE:.+]] = hal.devices.get %{{.+}}
  %device = stream.context.resolve : !hal.device
  // CHECK: util.return %[[DEVICE]]
  util.return %device : !hal.device
}

// -----

// CHECK-LABEL: @contextResolveDeviceQueueAffinityAny
util.func public @contextResolveDeviceQueueAffinityAny() -> (!hal.device, i64) {
  // CHECK-DAG: %[[DEVICE:.+]] = hal.devices.get %{{.+}}
  // CHECK-DAG: %[[QUEUE_AFFINITY:.+]] = arith.constant -1 : i64
  %device, %queue_affinity_any = stream.context.resolve on(#hal.affinity.queue<*>) : !hal.device, i64
  // CHECK: util.return %[[DEVICE]], %[[QUEUE_AFFINITY]]
  util.return %device, %queue_affinity_any : !hal.device, i64
}

// -----

// CHECK-LABEL: @contextResolveDeviceQueueAffinity45
util.func public @contextResolveDeviceQueueAffinity45() -> (!hal.device, i64) {
  // CHECK: %[[DEVICE:.+]] = hal.devices.get %{{.+}}
  // CHECK-DAG: %[[QUEUE_AFFINITY:.+]] = arith.constant 48 : i64
  %device, %queue_affinity_45 = stream.context.resolve on(#hal.affinity.queue<[4, 5]>) : !hal.device, i64
  // CHECK: util.return %[[DEVICE]], %[[QUEUE_AFFINITY]]
  util.return %device, %queue_affinity_45 : !hal.device, i64
}
