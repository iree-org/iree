// RUN: iree-opt --split-input-file --mlir-print-local-scope --iree-hal-annotate-target-devices %s | FileCheck %s

// CHECK: util.global private @unspecified_ordinal
// CHECK-SAME: hal.device.targets = []
util.global private @unspecified_ordinal = #hal.device.ordinal<0> : !hal.device
util.func private @unspecified_ordinal_load() -> !hal.device {
  // CHECK: util.global.load
  // CHECK-SAME-LITERAL: hal.device.targets.results = [[]]
  %device = util.global.load immutable @unspecified_ordinal : !hal.device
  util.return %device : !hal.device
}

// -----

// CHECK: util.global private @target_device
// CHECK-SAME-LITERAL: hal.device.targets = [#hal.device.target<"device">
util.global private @target_device = #hal.device.target<"device"> : !hal.device
util.func private @target_device_load() -> !hal.device {
  // CHECK: util.global.load
  // CHECK-SAME-LITERAL: hal.device.targets.results = [[#hal.device.target<"device"> : !hal.device]]
  %device = util.global.load immutable @target_device : !hal.device
  util.return %device : !hal.device
}

// -----

// Tests that fallbacks get propagated (this is not a real use of fallbacks,
// but tests the indirection works).

// CHECK: util.global private @required_device
// CHECK-SAME: hal.device.targets = [#hal.device.target<"required"> : !hal.device]
util.global private @required_device = #hal.device.target<"required"> : !hal.device
// CHECK: util.global private @optional_device
// CHECK-SAME-LITERAL: hal.device.targets = [#hal.device.target<"required"> : !hal.device]
util.global private @optional_device = #hal.device.fallback<@required_device> : !hal.device
util.func private @optional_device_load() -> !hal.device {
  // CHECK: util.global.load
  // CHECK-SAME-LITERAL: hal.device.targets.results = [[#hal.device.target<"required"> : !hal.device]]
  %device = util.global.load immutable @optional_device : !hal.device
  util.return %device : !hal.device
}

// -----

// CHECK: util.global private @selected_device
// CHECK-SAME-LITERAL: hal.device.targets = [[#hal.device.target<"a"> : !hal.device, #hal.device.target<"b"> : !hal.device]]
util.global private @selected_device = #hal.device.select<[
  #hal.device.target<"a"> : !hal.device,
  #hal.device.target<"b"> : !hal.device
]> : !hal.device
util.func private @selected_device_load() -> !hal.device {
  // CHECK: util.global.load
  // CHECK-SAME-LITERAL: hal.device.targets.results = [[#hal.device.target<"a"> : !hal.device, #hal.device.target<"b"> : !hal.device]]
  %device = util.global.load immutable @selected_device : !hal.device
  util.return %device : !hal.device
}

// -----

// Tests that both potential sets are propagated across select ops.

util.global private @device_a = #hal.device.target<"device_a"> : !hal.device
util.global private @device_b = #hal.device.target<"device_b"> : !hal.device
util.func private @arith_select(%cond: i1) -> !hal.device {
  %device_a = util.global.load immutable @device_a : !hal.device
  %device_b = util.global.load immutable @device_b : !hal.device
  // CHECK: arith.select
  // CHECK-SAME-LITERAL: hal.device.targets.operands = [[#hal.device.target<"device_a"> : !hal.device], [#hal.device.target<"device_b"> : !hal.device]]
  // CHECK-SAME-LITERAL: hal.device.targets.results = [[#hal.device.target<"device_a"> : !hal.device, #hal.device.target<"device_b"> : !hal.device]]
  %select = arith.select %cond, %device_a, %device_b : !hal.device
  util.return %select : !hal.device
}

// -----

// Tests that both potential sets are propagated across allocator select ops.

util.global private @device_a = #hal.device.target<"device_a"> : !hal.device
util.global private @device_b = #hal.device.target<"device_b"> : !hal.device
util.func private @allocator_select() -> !hal.device {
  %device_a = util.global.load immutable @device_a : !hal.device
  %affinity_a = arith.constant 100 : i64
  %device_b = util.global.load immutable @device_b : !hal.device
  %affinity_b = arith.constant 101 : i64
  %type = arith.constant 2 : i32
  %usage = arith.constant 3 : i32
  // CHECK: hal.allocator.select
  // CHECK-SAME-LITERAL: hal.device.targets.operands = [[#hal.device.target<"device_a"> : !hal.device], [#hal.device.target<"device_b"> : !hal.device]]
  // CHECK-SAME-LITERAL: hal.device.targets.results = [[#hal.device.target<"device_a"> : !hal.device, #hal.device.target<"device_b"> : !hal.device]]}
  %device, %queue_affinity = hal.allocator.select
      from([
        (%device_a, %affinity_a : !hal.device, i64),
        (%device_b, %affinity_b : !hal.device, i64)
      ])
      type(%type) usage(%usage) : !hal.device, i64
  util.return %device : !hal.device
}

// -----

// Tests that device analysis tracks across function calls.

util.global private @device_a = #hal.device.target<"device_a"> : !hal.device
util.global private @device_b = #hal.device.target<"device_b"> : !hal.device
// CHECK: util.func private @caller
util.func private @caller() -> !hal.device {
  %device_a = util.global.load immutable @device_a : !hal.device
  // CHECK: util.call @callee_a
  // CHECK-SAME-LITERAL: hal.device.targets.operands = [[#hal.device.target<"device_a"> : !hal.device]]
  // CHECK-SAME-LITERAL: hal.device.targets.results = [[#hal.device.target<"device_a"> : !hal.device]]
  %call_a = util.call @callee_a(%device_a) : (!hal.device) -> !hal.device
  // CHECK: util.call @callee_ab
  // CHECK-SAME-LITERAL: hal.device.targets.operands = [[#hal.device.target<"device_a"> : !hal.device]]
  // CHECK-SAME-LITERAL: hal.device.targets.results = [[#hal.device.target<"device_a"> : !hal.device, #hal.device.target<"device_b"> : !hal.device]]
  %call_ab = util.call @callee_ab(%call_a) : (!hal.device) -> !hal.device
  // CHECK: util.return
  // CHECK-SAME-LITERAL: hal.device.targets.operands = [[#hal.device.target<"device_a"> : !hal.device, #hal.device.target<"device_b"> : !hal.device]]
  util.return %call_ab : !hal.device
}
// CHECK: util.func private @callee_a
// CHECK-SAME-LITERAL: !hal.device {hal.device.targets = [#hal.device.target<"device_a"> : !hal.device]}
util.func private @callee_a(%device: !hal.device) -> !hal.device {
  // CHECK: util.return
  // CHECK-SAME-LITERAL: hal.device.targets.operands = [[#hal.device.target<"device_a"> : !hal.device]]
  util.return %device : !hal.device
}
// CHECK: util.func private @callee_ab
// CHECK-SAME-LITERAL: !hal.device {hal.device.targets = [#hal.device.target<"device_a"> : !hal.device]}
util.func private @callee_ab(%device_a: !hal.device) -> !hal.device {
  %cond = arith.constant true
  %device_b = util.global.load immutable @device_b : !hal.device
  // CHECK: arith.select
  // CHECK-SAME-LITERAL: hal.device.targets.operands = [[#hal.device.target<"device_a"> : !hal.device], [#hal.device.target<"device_b"> : !hal.device]]
  // CHECK-SAME-LITERAL: hal.device.targets.results = [[#hal.device.target<"device_a"> : !hal.device, #hal.device.target<"device_b"> : !hal.device]]
  %device_ab = arith.select %cond, %device_a, %device_b : !hal.device
  // CHECK: util.return
  // CHECK-SAME-LITERAL: hal.device.targets.operands = [[#hal.device.target<"device_a"> : !hal.device, #hal.device.target<"device_b"> : !hal.device]]
  util.return %device_ab : !hal.device
}
