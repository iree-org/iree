// RUN: iree-opt --split-input-file --iree-hal-memoize-device-queries --canonicalize %s | FileCheck %s

//      CHECK: util.global private @_device_query_0 : i1
// CHECK-NEXT: util.global private @_device_query_0_ok : i1
// CHECK-NEXT: util.initializer {
//  CHECK-DAG:   %[[DEVICE:.+]] = hal.devices.get %{{.+}}
// CHECK-NEXT:   %[[OK0:.+]], %[[VALUE0:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.device.id" :: "id0*") : i1, i1 = false
// CHECK-NEXT:   util.global.store %[[OK0]], @_device_query_0_ok : i1
// CHECK-NEXT:   util.global.store %[[VALUE0]], @_device_query_0 : i1

//      CHECK: util.global private @_device_query_1 : i1
// CHECK-NEXT: util.global private @_device_query_1_ok : i1
// CHECK-NEXT: util.initializer {
//  CHECK-DAG:   %[[DEVICE:.+]] = hal.devices.get %{{.+}}
// CHECK-NEXT:   %[[OK1:.+]], %[[VALUE1:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.device.id" :: "id1") : i1, i1 = false
// CHECK-NEXT:   util.global.store %[[OK1]], @_device_query_1_ok : i1
// CHECK-NEXT:   util.global.store %[[VALUE1]], @_device_query_1 : i1

// CHECK: util.global private @_device_query_2

// CHECK-LABEL: util.func public @device_matchers
util.func public @device_matchers(%device : !hal.device) -> (i1, i1, i1, i1, i1, i1) {
  // Same queries (same variables):
  // CHECK-NEXT: = util.global.load @_device_query_0_ok : i1
  // CHECK-NEXT: = util.global.load @_device_query_0 : i1
  %id0_a_ok, %id0_a = hal.device.query<%device : !hal.device> key("hal.device.id" :: "id0*") : i1, i1 = false
  // CHECK-NEXT: = util.global.load @_device_query_0_ok : i1
  // CHECK-NEXT: = util.global.load @_device_query_0 : i1
  %id0_b_ok, %id0_b = hal.device.query<%device : !hal.device> key("hal.device.id" :: "id0*") : i1, i1 = false

  // Same query but with different defaults (different variables):
  // CHECK-NEXT: = util.global.load @_device_query_1 : i1
  %id1_a_ok, %id1_a = hal.device.query<%device : !hal.device> key("hal.device.id" :: "id1") : i1, i1 = false
  // CHECK-NEXT: = util.global.load @_device_query_2 : i1
  %id1_b_ok, %id1_b = hal.device.query<%device : !hal.device> key("hal.device.id" :: "id1") : i1, i1 = true

  util.return %id0_a_ok, %id0_a, %id0_b_ok, %id0_b, %id1_a, %id1_b : i1, i1, i1, i1, i1, i1
}
