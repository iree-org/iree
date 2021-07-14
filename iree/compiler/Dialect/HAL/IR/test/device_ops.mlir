// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @device_allocator
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
func @device_allocator(%device: !hal.device) -> !hal.allocator {
  // CHECK: %allocator = hal.device.allocator<%[[DEVICE]] : !hal.device> : !hal.allocator
  %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
  return %allocator : !hal.allocator
}

// -----

// CHECK-LABEL: @device_switch
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
func @device_switch(%device: !hal.device) -> i32 {
  // CHECK-DAG: %[[C0:.+]] = constant 0
  %c0 = constant 0 : i32
  // CHECK-DAG: %[[C1:.+]] = constant 1
  %c1 = constant 1 : i32
  // CHECK-DAG: %[[C2:.+]] = constant 2
  %c2 = constant 2 : i32
  // CHECK: = hal.device.switch<%[[DEVICE]] : !hal.device> -> i32
  %0 = hal.device.switch<%device : !hal.device> -> i32
    // CHECK-NEXT: #hal.device.match.id<"vulkan-v1.?-*">(%[[C1A:.+]] = %[[C1]] : i32) {
    #hal.device.match.id<"vulkan-v1.?-*">(%c1a = %c1 : i32) {
      // CHECK-NEXT: hal.return %[[C1A]] : i32
      hal.return %c1a : i32
      // CHECK-NEXT: },
    },
    // CHECK-NEXT: #hal.match.any<[#hal.device.match.id<"vmvx">, #hal.device.match.id<"vulkan-*">]>(%[[C2A:.+]] = %[[C2]] : i32) {
    #hal.match.any<[#hal.device.match.id<"vmvx">, #hal.device.match.id<"vulkan-*">]>(%c2a = %c2 : i32) {
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
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
func @device_matchers(%device : !hal.device) -> i1 {
  // CHECK: = hal.device.match.id<%[[DEVICE]] : !hal.device> pattern("vulkan-*") : i1
  %0 = hal.device.match.id<%device : !hal.device> pattern("vulkan-*") : i1
  return %0 : i1
}

// -----

// CHECK-LABEL: @device_query
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
func @device_query(%device : !hal.device) -> (i1, i32) {
  // CHECK: = hal.device.query<%[[DEVICE]] : !hal.device> key("sys" :: "foo") : i1, i32
  %ok, %value = hal.device.query<%device : !hal.device> key("sys" :: "foo") : i1, i32
  return %ok, %value : i1, i32
}
