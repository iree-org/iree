// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | FileCheck %s

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
func @device_query(%device : !hal.device) -> (i1, i32) {
  // CHECK: = hal.device.query<%[[DEVICE]] : !hal.device> key("sys" :: "foo") : i1, i32
  %ok, %value = hal.device.query<%device : !hal.device> key("sys" :: "foo") : i1, i32
  return %ok, %value : i1, i32
}
