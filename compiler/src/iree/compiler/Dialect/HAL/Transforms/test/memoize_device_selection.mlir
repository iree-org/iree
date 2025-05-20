// RUN: iree-opt --split-input-file --iree-hal-memoize-device-selection %s | FileCheck %s

// Tests memoization of a select by hoisting it to an initializer prior to all
// queries.

util.global private @device_a : !hal.device
util.global private @device_b : !hal.device
util.global private @device_c : !hal.device

// CHECK: util.global private @__allocator_select_[[DEVICE0:.+]] : !hal.device
// CHECK: util.global private @__allocator_select_[[AFFINITY0:.+]] : i64
// CHECK: util.initializer
// CHECK:   %[[I0_DEVICE_A:.+]] = util.global.load @device_a
// CHECK:   %[[I0_AFFINITY_A:.+]] = arith.constant -1
// CHECK:   %[[I0_DEVICE_B:.+]] = util.global.load @device_b
// CHECK:   %[[I0_AFFINITY_B:.+]] = arith.constant -1
// CHECK:   %[[I0_TYPE:.+]] = arith.constant 70
// CHECK:   %[[I0_USAGE:.+]] = arith.constant 3
// CHECK:   %[[I0_DEVICE:.+]], %[[I0_AFFINITY:.+]] = hal.allocator.select from([
// CHECK:     (%[[I0_DEVICE_A]], %[[I0_AFFINITY_A]] : !hal.device, i64)
// CHECK:     (%[[I0_DEVICE_B]], %[[I0_AFFINITY_B]] : !hal.device, i64)
// CHECK:   ]) type(%[[I0_TYPE]]) usage(%[[I0_USAGE]])
// CHECK:   util.global.store %[[I0_DEVICE]], @__allocator_select_[[DEVICE0]]
// CHECK:   util.global.store %[[I0_AFFINITY]], @__allocator_select_[[AFFINITY0]]

// CHECK: @fn1
util.func public @fn1() -> (!hal.device, i64) {
  // CHECK-NOT: hal.allocator.select.attr
  // CHECK: %[[FN1_DEVICE:.+]] = util.global.load @__allocator_select_[[DEVICE0]]
  // CHECK: %[[FN1_AFFINITY:.+]] = util.global.load @__allocator_select_[[AFFINITY0]]
  %device, %queue_affinity = hal.allocator.select.attr
      from(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
      type(HostLocal) usage(Transfer) : !hal.device, i64
  // CHECK: util.return %[[FN1_DEVICE]], %[[FN1_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}

// CHECK: @fn2
util.func public @fn2() -> (!hal.device, i64) {
  // CHECK-NOT: hal.allocator.select.attr
  // CHECK: %[[FN2_DEVICE:.+]] = util.global.load @__allocator_select_[[DEVICE0]]
  // CHECK: %[[FN2_AFFINITY:.+]] = util.global.load @__allocator_select_[[AFFINITY0]]
  %device, %queue_affinity = hal.allocator.select.attr
      from(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
      type(HostLocal) usage(Transfer) : !hal.device, i64
  // CHECK: util.return %[[FN2_DEVICE]], %[[FN2_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}

// CHECK: util.global private @__allocator_select_[[DEVICE1:.+]] : !hal.device
// CHECK: util.global private @__allocator_select_[[AFFINITY1:.+]] : i64
// CHECK: util.initializer

// CHECK: @fn3
util.func public @fn3() -> (!hal.device, i64) {
  // CHECK-NOT: hal.allocator.select.attr
  // CHECK: %[[FN3_DEVICE:.+]] = util.global.load @__allocator_select_[[DEVICE1]]
  // CHECK: %[[FN3_AFFINITY:.+]] = util.global.load @__allocator_select_[[AFFINITY1]]
  %device, %queue_affinity = hal.allocator.select.attr
      from(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_c>]>)
      type(HostLocal) usage(DispatchStorage) : !hal.device, i64
  // CHECK: util.return %[[FN3_DEVICE]], %[[FN3_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}
