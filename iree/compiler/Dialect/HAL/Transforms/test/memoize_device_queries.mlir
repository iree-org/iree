// RUN: iree-opt -split-input-file -iree-hal-memoize-device-queries -canonicalize %s | IreeFileCheck %s

//      CHECK: hal.variable @_device_query_0 init(@_device_query_0_initializer) : i1
//      CHECK: func private @_device_query_0_initializer() -> i1
// CHECK-NEXT:   %[[DEVICE:.+]] = hal.ex.shared_device : !hal.device
// CHECK-NEXT:   %[[OK0:.+]], %[[VALUE0:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.device.id" :: "id0*") : i1, i1 = false
// CHECK-NEXT:   return %[[VALUE0]] : i1

//      CHECK: hal.variable @_device_query_1 init(@_device_query_1_initializer) : i1
//      CHECK: func private @_device_query_1_initializer() -> i1
// CHECK-NEXT:   %[[DEVICE:.+]] = hal.ex.shared_device : !hal.device
// CHECK-NEXT:   %[[OK1:.+]], %[[VALUE1:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.device.id" :: "id1") : i1, i1 = false
// CHECK-NEXT:   return %[[VALUE1]] : i1

// CHECK: hal.variable @_device_query_2

// CHECK-LABEL: func @device_matchers
func @device_matchers(%device : !hal.device) -> (i1, i1, i1, i1) {
  // Same queries (same variables):
  // CHECK-NEXT: = hal.variable.load @_device_query_0 : i1
  %id0_a_ok, %id0_a = hal.device.query<%device : !hal.device> key("hal.device.id" :: "id0*") : i1, i1 = false
  // CHECK-NEXT: = hal.variable.load @_device_query_0 : i1
  %id0_b_ok, %id0_b = hal.device.query<%device : !hal.device> key("hal.device.id" :: "id0*") : i1, i1 = false

  // Same query but with different defaults (different variables):
  // CHECK-NEXT: = hal.variable.load @_device_query_1 : i1
  %id1_a_ok, %id1_a = hal.device.query<%device : !hal.device> key("hal.device.id" :: "id1") : i1, i1 = false
  // CHECK-NEXT: = hal.variable.load @_device_query_2 : i1
  %id1_b_ok, %id1_b = hal.device.query<%device : !hal.device> key("hal.device.id" :: "id1") : i1, i1 = true

  return %id0_a, %id0_b, %id1_a, %id1_b : i1, i1, i1, i1
}
