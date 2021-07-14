// RUN: iree-opt -split-input-file -iree-hal-memoize-device-queries -canonicalize %s | IreeFileCheck %s

//      CHECK: hal.variable @_device_query_0 init(@_device_query_0_initializer) : i1
//      CHECK: func private @_device_query_0_initializer() -> i1
// CHECK-NEXT:   %[[DEVICE:.+]] = hal.ex.shared_device : !hal.device
// CHECK-NEXT:   %[[OK:.+]], %[[VALUE:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.device.id" :: "id0*") : i1, i1 = false
// CHECK-NEXT:   return %[[VALUE]] : i1

// CHECK: hal.variable @_device_query_1
// CHECK: hal.variable @_device_query_2
// CHECK: hal.variable @_device_query_3

// CHECK-LABEL: func @device_matchers
func @device_matchers(%device : !hal.device) -> (i1, i1, i1, i1, i1, i1) {
  // Same queries (same variables):
  // CHECK-NEXT: = hal.variable.load @_device_query_0 : i1
  %id0_a = hal.device.match.id<%device : !hal.device> pattern("id0*") : i1
  // CHECK-NEXT: = hal.variable.load @_device_query_0 : i1
  %id0_b = hal.device.match.id<%device : !hal.device> pattern("id0*") : i1

  // Same query but with different defaults (different variables):
  // CHECK-NEXT: = hal.variable.load @_device_query_1 : i1
  %id1_a_ok, %id1_a = hal.device.query<%device : !hal.device> key("hal.device.id" :: "id1") : i1, i1 = false
  // CHECK-NEXT: = hal.variable.load @_device_query_2 : i1
  %id1_b_ok, %id1_b = hal.device.query<%device : !hal.device> key("hal.device.id" :: "id1") : i1, i1 = true

  // Same query via both unexpanded and expanded forms (same variables):
  // CHECK-NEXT: = hal.variable.load @_device_query_3 : i1
  %id2_a = hal.device.match.id<%device : !hal.device> pattern("id2*") : i1
  // CHECK-NEXT: = hal.variable.load @_device_query_3 : i1
  %id2_ok, %id2_b = hal.device.query<%device : !hal.device> key("hal.device.id" :: "id2*") : i1, i1 = false

  return %id0_a, %id0_b, %id1_a, %id1_b, %id2_a, %id2_b : i1, i1, i1, i1, i1, i1
}
