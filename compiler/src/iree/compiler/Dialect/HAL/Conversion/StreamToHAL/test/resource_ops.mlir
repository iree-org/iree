// RUN: iree-opt --split-input-file --iree-hal-conversion %s | FileCheck %s

util.global private @device : !hal.device

// CHECK-LABEL: @resourceAlloc
util.func public @resourceAlloc(%arg0: index) -> !stream.resource<transient> {
  // CHECK: %[[RET0:.+]] = hal.allocator.allocate
  // CHECK-SAME: type("DeviceVisible|DeviceLocal")
  // CHECK-SAME: usage("{{.+}}Transfer{{.+}}Dispatch{{.+}}")
  // CHECK-SAME: : !hal.buffer{%arg0}
  %0 = stream.resource.alloc uninitialized on(#hal.device.affinity<@device>) : !stream.resource<transient>{%arg0}
  // CHECK: util.return %[[RET0]]
  util.return %0 : !stream.resource<transient>
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @resourceAlloca
// CHECK-SAME: (%[[SIZE:.+]]: index)
util.func public @resourceAlloca(%size: index) -> (!stream.resource<transient>, !stream.timepoint) {
  // CHECK: %[[WAIT_FENCE:.+]] = util.null : !hal.fence
  // CHECK: %[[SIGNAL_FENCE:.+]] = hal.fence.create
  // CHECK: %[[RET0:.+]] = hal.device.queue.alloca
  // CHECK-SAME: affinity(%c-1
  // CHECK-SAME: wait(%[[WAIT_FENCE]])
  // CHECK-SAME: signal(%[[SIGNAL_FENCE]])
  // CHECK-SAME: pool(%c0
  // CHECK-SAME: type("DeviceVisible|DeviceLocal")
  // CHECK-SAME: usage("{{.+}}Transfer{{.+}}DispatchStorage")
  // CHECK-SAME: flags("None")
  // CHECK-SAME: : !hal.buffer{%[[SIZE]]}
  %0:2 = stream.resource.alloca uninitialized on(#hal.device.affinity<@device>) : !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: util.return %[[RET0]], %[[SIGNAL_FENCE]]
  util.return %0#0, %0#1 : !stream.resource<transient>, !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @resourceAllocaAwait
// CHECK-SAME: (%[[SIZE:.+]]: index, %[[WAIT_FENCE:.+]]: !hal.fence)
util.func public @resourceAllocaAwait(%size: index, %await_timepoint: !stream.timepoint) -> (!stream.resource<transient>, !stream.timepoint) {
  // CHECK: %[[SIGNAL_FENCE:.+]] = hal.fence.create
  // CHECK: %[[RET0:.+]] = hal.device.queue.alloca
  // CHECK-SAME: affinity(%c-1
  // CHECK-SAME: wait(%[[WAIT_FENCE]])
  // CHECK-SAME: signal(%[[SIGNAL_FENCE]])
  // CHECK-SAME: pool(%c0
  // CHECK-SAME: type("DeviceVisible|DeviceLocal")
  // CHECK-SAME: usage("{{.+}}Transfer{{.+}}DispatchStorage")
  // CHECK-SAME: flags("None")
  // CHECK-SAME: : !hal.buffer{%[[SIZE]]}
  %0:2 = stream.resource.alloca uninitialized on(#hal.device.affinity<@device>) await(%await_timepoint) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: util.return %[[RET0]], %[[SIGNAL_FENCE]]
  util.return %0#0, %0#1 : !stream.resource<transient>, !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @resourceDealloca
// CHECK-SAME: (%[[SIZE:.+]]: index, %[[RESOURCE:.+]]: !hal.buffer)
util.func public @resourceDealloca(%size: index, %resource: !stream.resource<transient>) -> !stream.timepoint {
  // CHECK: %[[WAIT_FENCE:.+]] = util.null : !hal.fence
  // CHECK: %[[SIGNAL_FENCE:.+]] = hal.fence.create
  // CHECK: hal.device.queue.dealloca
  // CHECK-SAME: affinity(%c-1
  // CHECK-SAME: wait(%[[WAIT_FENCE]])
  // CHECK-SAME: signal(%[[SIGNAL_FENCE]])
  // CHECK-SAME: buffer(%[[RESOURCE]] : !hal.buffer)
  %0 = stream.resource.dealloca on(#hal.device.affinity<@device>) %resource : !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: util.return %[[SIGNAL_FENCE]]
  util.return %0 : !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @resourceDeallocaAwait
// CHECK-SAME: (%[[SIZE:.+]]: index, %[[RESOURCE:.+]]: !hal.buffer, %[[WAIT_FENCE:.+]]: !hal.fence)
util.func public @resourceDeallocaAwait(%size: index, %resource: !stream.resource<transient>, %await_timepoint: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[SIGNAL_FENCE:.+]] = hal.fence.create
  // CHECK: hal.device.queue.dealloca
  // CHECK-SAME: affinity(%c-1
  // CHECK-SAME: wait(%[[WAIT_FENCE]])
  // CHECK-SAME: signal(%[[SIGNAL_FENCE]])
  // CHECK-SAME: buffer(%[[RESOURCE]] : !hal.buffer)
  %0 = stream.resource.dealloca on(#hal.device.affinity<@device>) await(%await_timepoint) => %resource : !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: util.return %[[SIGNAL_FENCE]]
  util.return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @resourceRetain
util.func private @resourceRetain(%arg0: !stream.resource<transient>, %arg1: index) {
  // CHECK: hal.buffer.allocation.preserve<%arg0 : !hal.buffer>
  stream.resource.retain %arg0 : !stream.resource<transient>{%arg1}
  util.return
}

// -----

// CHECK-LABEL: @resourceRelease
util.func private @resourceRelease(%arg0: !stream.resource<transient>, %arg1: index) -> i1 {
  // CHECK: %[[WAS_TERMINAL:.+]] = hal.buffer.allocation.discard<%arg0 : !hal.buffer>
  %was_terminal = stream.resource.release %arg0 : !stream.resource<transient>{%arg1}
  // CHECK: util.return %[[WAS_TERMINAL]]
  util.return %was_terminal : i1
}

// -----

// CHECK-LABEL: @resourceIsTerminal
util.func private @resourceIsTerminal(%arg0: !stream.resource<transient>, %arg1: index) -> i1 {
  // CHECK: %[[IS_TERMINAL:.+]] = hal.buffer.allocation.is_terminal<%arg0 : !hal.buffer>
  %is_terminal = stream.resource.is_terminal %arg0 : !stream.resource<transient>{%arg1}
  // CHECK: util.return %[[IS_TERMINAL]]
  util.return %is_terminal : i1
}

// -----

// CHECK-LABEL: @resourceSize
util.func public @resourceSize(%arg0: !stream.resource<transient>) -> index {
  // CHECK: %[[SIZE:.+]] = hal.buffer.length<%arg0 : !hal.buffer> : index
  %0 = stream.resource.size %arg0 : !stream.resource<transient>
  // CHECK: util.return %[[SIZE]]
  util.return %0 : index
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @resourceTryMap
util.func public @resourceTryMap(%arg0: !util.buffer) -> (i1, !stream.resource<constant>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: %[[DID_IMPORT:.+]], %[[IMPORTED:.+]] = hal.allocator.import
  // CHECK-SAME: source(%arg0 : !util.buffer)[%c0, %c128]
  // CHECK-SAME: type("DeviceVisible|DeviceLocal")
  // CHECK-SAME: usage("{{.+}}Transfer{{.+}}Dispatch{{.+}}SharingImmutable") : i1, !hal.
  %did_map, %mapping = stream.resource.try_map on(#hal.device.affinity<@device>) %arg0[%c0] : !util.buffer -> i1, !stream.resource<constant>{%c128}
  // CHECK: util.return %[[DID_IMPORT]], %[[IMPORTED]]
  util.return %did_map, %mapping : i1, !stream.resource<constant>
}

// -----

// CHECK-LABEL: @resourceLoad
util.func public @resourceLoad(%arg0: !stream.resource<staging>, %arg1: index) -> i32 {
  %c4 = arith.constant 4 : index
  // CHECK: %[[RET0:.+]] = hal.buffer.load<%arg0 : !hal.buffer>[%c4] : i32
  %0 = stream.resource.load %arg0[%c4] : !stream.resource<staging>{%arg1} -> i32
  // CHECK: util.return %[[RET0]]
  util.return %0 : i32
}

// -----

// CHECK-LABEL: @resourceStore
util.func public @resourceStore(%arg0: !stream.resource<staging>, %arg1: index) {
  %c4 = arith.constant 4 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: hal.buffer.store<%arg0 : !hal.buffer>[%c4] value(%c123_i32 : i32)
  stream.resource.store %c123_i32, %arg0[%c4] : i32 -> !stream.resource<staging>{%arg1}
  util.return
}

// -----

// CHECK-LABEL: @resourceSubview
util.func public @resourceSubview(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.resource<transient> {
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  // CHECK: %[[RET0:.+]] = hal.buffer.subspan<%arg0 : !hal.buffer>[%c128, %c256] : !hal.buffer
  %0 = stream.resource.subview %arg0[%c128] : !stream.resource<transient>{%arg1} -> !stream.resource<transient>{%c256}
  // CHECK: util.return %[[RET0]]
  util.return %0 : !stream.resource<transient>
}
