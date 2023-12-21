// RUN: iree-opt --split-input-file --allow-unregistered-dialect --iree-hal-conversion %s | FileCheck %s

// CHECK-LABEL: @contextResolveAllocator
func.func @contextResolveAllocator() -> !hal.allocator {
  // CHECK: %[[DEVICE:.+]] = hal.devices.get %{{.+}}
  // CHECK: %[[ALLOCATOR:.+]] = hal.device.allocator<%[[DEVICE]] : !hal.device> : !hal.allocator
  %allocator = stream.context.resolve : !hal.allocator
  // CHECK: return %[[ALLOCATOR]]
  return %allocator : !hal.allocator
}

// -----

// CHECK-LABEL: @contextResolveDevice
func.func @contextResolveDevice() -> !hal.device {
  // CHECK: %[[DEVICE:.+]] = hal.devices.get %{{.+}}
  %device = stream.context.resolve : !hal.device
  // CHECK: return %[[DEVICE]]
  return %device : !hal.device
}

// -----

// CHECK-LABEL: @contextResolveDeviceQueueAffinityAny
func.func @contextResolveDeviceQueueAffinityAny() -> (!hal.device, i64) {
  // CHECK-DAG: %[[DEVICE:.+]] = hal.devices.get %{{.+}}
  // CHECK-DAG: %[[QUEUE_AFFINITY:.+]] = arith.constant -1 : i64
  %device, %queue_affinity_any = stream.context.resolve on(#hal.affinity.queue<*>) : !hal.device, i64
  // CHECK: return %[[DEVICE]], %[[QUEUE_AFFINITY]]
  return %device, %queue_affinity_any : !hal.device, i64
}

// -----

// CHECK-LABEL: @contextResolveDeviceQueueAffinity45
func.func @contextResolveDeviceQueueAffinity45() -> (!hal.device, i64) {
  // CHECK: %[[DEVICE:.+]] = hal.devices.get %{{.+}}
  // CHECK-DAG: %[[QUEUE_AFFINITY:.+]] = arith.constant 48 : i64
  %device, %queue_affinity_45 = stream.context.resolve on(#hal.affinity.queue<[4, 5]>) : !hal.device, i64
  // CHECK: return %[[DEVICE]], %[[QUEUE_AFFINITY]]
  return %device, %queue_affinity_45 : !hal.device, i64
}
