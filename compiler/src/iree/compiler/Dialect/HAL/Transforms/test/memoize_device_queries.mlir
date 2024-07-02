// RUN: iree-opt --split-input-file --iree-hal-memoize-device-queries --canonicalize %s | FileCheck %s

// Tests that unknown devices (here passed as an arg on a public function) don't
// get memoized.

// CHECK-LABEL: @unknown_device
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
util.func public @unknown_device(%device: !hal.device) -> i1 {
  // CHECK-NEXT: hal.device.query<%[[DEVICE]]
  %id0_ok, %id0 = hal.device.query<%device : !hal.device> key("hal.device.id" :: "id0") : i1, i1 = false
  util.return %id0 : i1
}

// -----

// Tests that multiple possible devices disable memoization.
// TODO(multi-device): enable propagation of queried values across the program.
// We should be able to track back to each global, memoize there, then pass
// through the value as a normal SSA value.

util.global private @device_a : !hal.device
util.global private @device_b : !hal.device

// CHECK-LABEL: @multi_device_not_memoized
util.func public @multi_device_not_memoized(%cond: i1) -> i1 {
  // CHECK-DAG: %[[DEVICE_A:.+]] = util.global.load @device_a
  %device_a = util.global.load @device_a : !hal.device
  // CHECK-DAG: %[[DEVICE_B:.+]] = util.global.load @device_b
  %device_b = util.global.load @device_b : !hal.device
  // CHECK: %[[DEVICE_AB:.+]] = arith.select %{{.+}}, %[[DEVICE_A]], %[[DEVICE_B]]
  %device_ab = arith.select %cond, %device_a, %device_b : !hal.device
  // CHECK-NEXT: hal.device.query<%[[DEVICE_AB]]
  %id0_ok, %id0 = hal.device.query<%device_ab : !hal.device> key("hal.device.id" :: "id0") : i1, i1 = false
  util.return %id0 : i1
}

// -----

// Tests basic hoisting of device queries up to an initializer per device.

// CHECK: util.global private @device
util.global private @device : !hal.device
// CHECK-NEXT: util.global private @__device_query_0_hal_device_id_id0_ok : i1
// CHECK-NEXT: util.global private @__device_query_0_hal_device_id_id0 : i1
// CHECK-NEXT: util.global private @__device_query_1_hal_device_id_id1_ok : i1
// CHECK-NEXT: util.global private @__device_query_1_hal_device_id_id1 : i1
// CHECK-NEXT: util.initializer
// CHECK: %[[DEVICE:.+]] = util.global.load @device : !hal.device
// CHECK: %[[OK0:.+]], %[[VALUE0:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.device.id" :: "id0") : i1, i1 = false
// CHECK: util.global.store %[[OK0]], @__device_query_0_hal_device_id_id0_ok : i1
// CHECK: util.global.store %[[VALUE0]], @__device_query_0_hal_device_id_id0 : i1
// CHECK: %[[OK1:.+]], %[[VALUE1:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.device.id" :: "id1") : i1, i1 = false
// CHECK: util.global.store %[[OK1]], @__device_query_1_hal_device_id_id1_ok : i1
// CHECK: util.global.store %[[VALUE1]], @__device_query_1_hal_device_id_id1 : i1

// CHECK: @single_device_memoized_0
util.func public @single_device_memoized_0() -> (i1, i1) {
  %device = util.global.load @device : !hal.device
  // CHECK-NEXT: = util.global.load @__device_query_0_hal_device_id_id0_ok : i1
  // CHECK-NEXT: = util.global.load @__device_query_0_hal_device_id_id0 : i1
  %id0_ok, %id0 = hal.device.query<%device : !hal.device> key("hal.device.id" :: "id0") : i1, i1 = false
  util.return %id0_ok, %id0 : i1, i1
}
// CHECK: @single_device_memoized_1
util.func public @single_device_memoized_1() -> (i1, i1) {
  %device = util.global.load @device : !hal.device
  // CHECK-NEXT: = util.global.load @__device_query_1_hal_device_id_id1_ok : i1
  // CHECK-NEXT: = util.global.load @__device_query_1_hal_device_id_id1 : i1
  %id1_ok, %id1 = hal.device.query<%device : !hal.device> key("hal.device.id" :: "id1") : i1, i1 = false
  util.return %id1_ok, %id1 : i1, i1
}
