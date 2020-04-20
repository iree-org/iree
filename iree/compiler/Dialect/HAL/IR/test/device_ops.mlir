// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @device_allocator
func @device_allocator() -> !hal.allocator {
  %0 = "test_hal.device"() : () -> !hal.device
  // CHECK: %allocator = hal.device.allocator %0 : !hal.allocator
  %allocator = hal.device.allocator %0 : !hal.allocator
  return %allocator : !hal.allocator
}

// -----

// CHECK-LABEL: @device_switch
func @device_switch() -> i32 {
  // CHECK-DAG: %[[C0:.+]] = constant 0
  %c0 = constant 0 : i32
  // CHECK-DAG: %[[C1:.+]] = constant 1
  %c1 = constant 1 : i32
  // CHECK-DAG: %[[C2:.+]] = constant 2
  %c2 = constant 2 : i32
  // CHECK-DAG: %[[DEVICE:.+]] = "test_hal.device"
  %device = "test_hal.device"() : () -> !hal.device
  // CHECK: = hal.device.switch(%[[DEVICE]] : !hal.device) -> i32
  %0 = hal.device.switch(%device : !hal.device) -> i32
    // CHECK-NEXT: #hal.device.match.id<"vulkan-v1.?-*">(%[[C1A:.+]] = %[[C1]] : i32) {
    #hal.device.match.id<"vulkan-v1.?-*">(%c1a = %c1 : i32) {
      // CHECK-NEXT: hal.return %[[C1A]] : i32
      hal.return %c1a : i32
      // CHECK-NEXT: },
    },
    // CHECK-NEXT: #hal.match.any<[#hal.device.match.id<"vmla">, #hal.device.match.id<"vulkan-*">]>(%[[C2A:.+]] = %[[C2]] : i32) {
    #hal.match.any<[#hal.device.match.id<"vmla">, #hal.device.match.id<"vulkan-*">]>(%c2a = %c2 : i32) {
      // CHECK-NEXT: hal.return %[[C2A]] : i32
      hal.return %c2a : i32
      // CHECK-NEXT: },
    },
    // CHECK-NEXT: #hal.match.always(%[[C0A:.+]] = %[[C0]] : i32) {
    #hal.match.always(%c0a = %c0 : i32) {
      // CHECK-NEXT: hal.return %[[C0A]] : i32
      hal.return %c0a : i32
      // CHECK-NEXT: }
    }
  return %0 : i32
}

// -----

// CHECK-LABEL: @device_matchers
// CHECK-SAME: %[[DEVICE:.+]]: !hal.device
func @device_matchers(%device : !hal.device) -> i1 {
  // CHECK: = hal.device.match.id %[[DEVICE]], pattern = ["vulkan-*"] : (!hal.device) -> i1
  %0 = hal.device.match.id %device, pattern = ["vulkan-*"] : (!hal.device) -> i1
  return %0 : i1
}
