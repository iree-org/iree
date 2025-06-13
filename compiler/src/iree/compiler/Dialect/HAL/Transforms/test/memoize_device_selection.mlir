// RUN: iree-opt --split-input-file --iree-hal-memoize-device-selection  %s | FileCheck %s

// Tests memoization of a select by hoisting it to an initializer prior to all
// queries.

util.global private @device_a : !hal.device
util.global private @device_b : !hal.device
util.global private @device_c : !hal.device

// CHECK: util.global private @__allocator_select_device[[DEVICE_0:.*]] : !hal.device
// CHECK: util.global private @__allocator_select_affinity[[AFFINITY_0:.*]] : i64
// CHECK: util.initializer
// CHECK:   %[[I0_DEVICE_A:.+]] = util.global.load @device_a
// CHECK:   %[[I0_AFFINITY_A:.+]] = arith.constant -1
// CHECK:   %[[I0_DEVICE_B:.+]] = util.global.load @device_b
// CHECK:   %[[I0_AFFINITY_B:.+]] = arith.constant -1
// CHECK:   %[[I0_MEMORY_TYPE:.+]] = hal.memory_type<{{.+}}Host{{.+}}> : i32
// CHECK:   %[[I0_BUFFER_USAGE:.+]] = hal.buffer_usage<{{.+}}Transfer{{.+}}> : i32
// CHECK:   %[[I0_DEVICE:.+]], %[[I0_AFFINITY:.+]] = hal.allocator.select from([
// CHECK:     (%[[I0_DEVICE_A]], %[[I0_AFFINITY_A]] : !hal.device, i64),
// CHECK:     (%[[I0_DEVICE_B]], %[[I0_AFFINITY_B]] : !hal.device, i64)
// CHECK:   ]) type(%[[I0_MEMORY_TYPE]]) usage(%[[I0_BUFFER_USAGE]])
// CHECK:   util.global.store %[[I0_DEVICE]], @__allocator_select_device[[DEVICE_0]]
// CHECK:   util.global.store %[[I0_AFFINITY]], @__allocator_select_affinity[[AFFINITY_0]]

// CHECK: @fn1
util.func public @fn1() -> (!hal.device, i64) {
  // CHECK-NOT: hal.allocator.select
  // CHECK: %[[FN1_DEVICE:.+]] = util.global.load @__allocator_select_device[[DEVICE_0]]
  // CHECK: %[[FN1_AFFINITY:.+]] = util.global.load @__allocator_select_affinity[[AFFINITY_0]]
  %memory_type = hal.memory_type<"HostLocal"> : i32
  %buffer_usage = hal.buffer_usage<"Transfer"> : i32
  %device_a = util.global.load @device_a : !hal.device
  %affinity_a = arith.constant -1 : i64
  %device_b = util.global.load @device_b : !hal.device
  %affinity_b = arith.constant -1 : i64
  %device, %queue_affinity = hal.allocator.select from([
    (%device_a, %affinity_a : !hal.device, i64),
    (%device_b, %affinity_b : !hal.device, i64)
  ]) type(%memory_type) usage(%buffer_usage) : !hal.device, i64
  // CHECK: util.return %[[FN1_DEVICE]], %[[FN1_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}

// CHECK: @fn2
util.func public @fn2() -> (!hal.device, i64) {
  // CHECK-NOT: hal.allocator.select
  // CHECK: %[[FN2_DEVICE:.+]] = util.global.load @__allocator_select_device[[DEVICE_0]]
  // CHECK: %[[FN2_AFFINITY:.+]] = util.global.load @__allocator_select_affinity[[AFFINITY_0]]
  %memory_type = hal.memory_type<"HostLocal"> : i32
  %buffer_usage = hal.buffer_usage<"Transfer"> : i32
  %device_a = util.global.load @device_a : !hal.device
  %affinity_a = arith.constant -1 : i64
  %device_b = util.global.load @device_b : !hal.device
  %affinity_b = arith.constant -1 : i64
  %device, %queue_affinity = hal.allocator.select from([
    (%device_a, %affinity_a : !hal.device, i64),
    (%device_b, %affinity_b : !hal.device, i64)
  ]) type(%memory_type) usage(%buffer_usage) : !hal.device, i64
  // CHECK: util.return %[[FN2_DEVICE]], %[[FN2_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}

// CHECK: util.global private @__allocator_select_device[[DEVICE_1:.*]] : !hal.device
// CHECK: util.global private @__allocator_select_affinity[[AFFINITY_1:.*]] : i64
// CHECK: util.initializer

// CHECK: @fn3
util.func public @fn3() -> (!hal.device, i64) {
  // CHECK-NOT: hal.allocator.select
  // CHECK: %[[FN3_DEVICE:.+]] = util.global.load @__allocator_select_device[[DEVICE_1]]
  // CHECK: %[[FN3_AFFINITY:.+]] = util.global.load @__allocator_select_affinity[[AFFINITY_1]]
  %memory_type = hal.memory_type<"HostLocal"> : i32
  %buffer_usage = hal.buffer_usage<"DispatchStorage"> : i32
  %device_a = util.global.load @device_a : !hal.device
  %affinity_a = arith.constant -1 : i64
  %device_c = util.global.load @device_c : !hal.device
  %affinity_c = arith.constant -1 : i64
  %device, %queue_affinity = hal.allocator.select from([
    (%device_a, %affinity_a : !hal.device, i64),
    (%device_c, %affinity_c : !hal.device, i64)
  ]) type(%memory_type) usage(%buffer_usage) : !hal.device, i64
  // CHECK: util.return %[[FN3_DEVICE]], %[[FN3_AFFINITY]]
  util.return %device, %queue_affinity : !hal.device, i64
}

// CHECK: @fn4
util.func public @fn4(%memory_type: i32, %buffer_usage: i32) -> (!hal.device, i64) {
  %device_a = util.global.load @device_a : !hal.device
  %affinity_a = arith.constant -1 : i64
  %device_b = util.global.load @device_b : !hal.device
  %affinity_b = arith.constant -1 : i64
  // should not get hoisted since they type/usage are not constants
  // CHECK: hal.allocator.select
  %device, %queue_affinity = hal.allocator.select from([
    (%device_a, %affinity_a : !hal.device, i64),
    (%device_b, %affinity_b : !hal.device, i64)
  ]) type(%memory_type) usage(%buffer_usage) : !hal.device, i64
  util.return %device, %queue_affinity : !hal.device, i64
}
