// RUN: iree-opt -split-input-file -iree-hal-memoize-device-queries %s | IreeFileCheck %s

//      CHECK: hal.variable @_device_match_id_0 init(@_device_match_id_0_initializer) : i1
//      CHECK: func @_device_match_id_0_initializer() -> i1
// CHECK-NEXT:   %[[DEVICE:.+]] = hal.ex.shared_device : !hal.device
// CHECK-NEXT:   %[[IS_MATCH:.+]] = hal.device.match.id %[[DEVICE]], pattern = ["vulkan-v1.?-*"] : (!hal.device) -> i1
// CHECK-NEXT:   return %[[IS_MATCH]] : i1

// CHECK: hal.variable @_device_match_id_1
// CHECK: hal.variable @_device_match_id_2

// CHECK-LABEL: func @device_matchers
func @device_matchers(%device : !hal.device) {
  // CHECK-NEXT: = hal.variable.load @_device_match_id_0 : i1
  %0 = hal.device.match.id %device, pattern = ["vulkan-v1.?-*"] : (!hal.device) -> i1
  // CHECK-NEXT: = hal.variable.load @_device_match_id_0 : i1
  %1 = hal.device.match.id %device, pattern = ["vulkan-v1.?-*"] : (!hal.device) -> i1
  // CHECK-NEXT: = hal.variable.load @_device_match_id_1 : i1
  %2 = hal.device.match.id %device, pattern = ["vulkan-v2.?-*"] : (!hal.device) -> i1
  // CHECK-NEXT: = hal.variable.load @_device_match_id_2 : i1
  %3 = hal.device.match.id %device, pattern = ["vulkan-*"] : (!hal.device) -> i1
  return
}
