// RUN: iree-opt --split-input-file --iree-hal-inline-conversion %s | FileCheck %s

// CHECK-LABEL: @resourceAlloc
// CHECK-SAME: (%[[LENGTH:.+]]: index)
util.func public @resourceAlloc(%length: index) -> !stream.resource<transient> {
  // CHECK: %[[BUFFER:.+]], %[[STORAGE:.+]] = hal_inline.buffer.allocate alignment(%c64) : !hal.buffer{%[[LENGTH]]}
  %result = stream.resource.alloc uninitialized : !stream.resource<transient>{%length}
  // CHECK: return %[[STORAGE]]
  util.return %result : !stream.resource<transient>
}

// -----

// CHECK-LABEL: @resourceAlloca
// CHECK-SAME: (%[[LENGTH:.+]]: index)
util.func public @resourceAlloca(%length: index) -> (!stream.resource<staging>, !stream.timepoint) {
  // CHECK: %[[BUFFER:.+]], %[[STORAGE:.+]] = hal_inline.buffer.allocate alignment(%c64) : !hal.buffer{%[[LENGTH]]}
  %0:2 = stream.resource.alloca uninitialized : !stream.resource<staging>{%length} => !stream.timepoint
  // CHECK: %[[IMMEDIATE:.+]] = arith.constant 0 : i64
  // CHECK: return %[[STORAGE]], %[[IMMEDIATE]]
  util.return %0#0, %0#1 : !stream.resource<staging>, !stream.timepoint
}

// -----

// CHECK-LABEL: @resourceAllocaAwait
// CHECK-SAME: (%[[LENGTH:.+]]: index, %[[TIMEPOINT:.+]]: i64)
util.func public @resourceAllocaAwait(%length: index, %await_timepoint: !stream.timepoint) -> (!stream.resource<staging>, !stream.timepoint) {
  // CHECK: %[[BUFFER:.+]], %[[STORAGE:.+]] = hal_inline.buffer.allocate alignment(%c64) : !hal.buffer{%[[LENGTH]]}
  %0:2 = stream.resource.alloca uninitialized await(%await_timepoint) => !stream.resource<staging>{%length} => !stream.timepoint
  // CHECK: %[[IMMEDIATE:.+]] = arith.constant 0 : i64
  // CHECK: return %[[STORAGE]], %[[IMMEDIATE]]
  util.return %0#0, %0#1 : !stream.resource<staging>, !stream.timepoint
}

// -----

// NOTE: we don't do anything with deallocs today but could add a discard op.

// CHECK-LABEL: @resourceDealloca
util.func public @resourceDealloca(%arg0: index, %arg1: !stream.resource<staging>, %arg2: !stream.timepoint) -> !stream.timepoint {
  %0 = stream.resource.dealloca %arg1 : !stream.resource<staging>{%arg0} => !stream.timepoint
  // CHECK: %[[IMMEDIATE:.+]] = arith.constant 0 : i64
  // CHECK: return %[[IMMEDIATE]]
  util.return %0 : !stream.timepoint
}

// -----

// NOTE: we don't do anything with deallocs today but could add a discard op.

// CHECK-LABEL: @resourceDeallocaAwait
util.func public @resourceDeallocaAwait(%arg0: index, %arg1: !stream.resource<staging>, %arg2: !stream.timepoint) -> !stream.timepoint {
  %0 = stream.resource.dealloca await(%arg2) => %arg1 : !stream.resource<staging>{%arg0} => !stream.timepoint
  // CHECK: %[[IMMEDIATE:.+]] = arith.constant 0 : i64
  // CHECK: return %[[IMMEDIATE]]
  util.return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @resourceSize
util.func public @resourceSize(%arg0: !stream.resource<transient>) -> index {
  // CHECK: %[[SIZE:.+]] = util.buffer.size %arg0
  %0 = stream.resource.size %arg0 : !stream.resource<transient>
  // CHECK: return %[[SIZE]]
  util.return %0 : index
}

// -----

// CHECK-LABEL: @resourceTryMap
// CHECK-SAME: (%[[SOURCE:.+]]: !util.buffer)
util.func public @resourceTryMap(%source: !util.buffer) -> (i1, !stream.resource<constant>) {
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 100
  %offset = arith.constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 128
  %length = arith.constant 128 : index
  // CHECK: %[[SOURCE_SIZE:.+]] = util.buffer.size %[[SOURCE]] : !util.buffer
  // CHECK: %[[MAPPING:.+]] = util.buffer.subspan %[[SOURCE]][%[[OFFSET]]] : !util.buffer{%[[SOURCE_SIZE]]} -> !util.buffer{%[[LENGTH]]}
  // CHECK-DAG: %[[DID_MAP:.+]] = arith.constant true
  %did_map, %mapping = stream.resource.try_map %source[%offset] : !util.buffer -> i1, !stream.resource<constant>{%length}
  // CHECK: return %[[DID_MAP]], %[[MAPPING]]
  util.return %did_map, %mapping : i1, !stream.resource<constant>
}

// -----

// CHECK-LABEL: @resourceLoad
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer, %[[BUFFER_SIZE:.+]]: index, %[[OFFSET:.+]]: index)
util.func public @resourceLoad(%resource: !stream.resource<staging>, %resource_size: index, %offset: index) -> i32 {
  // CHECK: %[[VALUE:.+]] = util.buffer.load %[[BUFFER]][%[[OFFSET]] for {{.+}}] : !util.buffer{%[[BUFFER_SIZE]]} -> i32
  %0 = stream.resource.load %resource[%offset] : !stream.resource<staging>{%resource_size} -> i32
  // CHECK: return %[[VALUE]]
  util.return %0 : i32
}

// -----

// CHECK-LABEL: @resourceStore
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer, %[[BUFFER_SIZE:.+]]: index, %[[OFFSET:.+]]: index)
util.func public @resourceStore(%resource: !stream.resource<staging>, %resource_size: index, %offset: index) {
  // CHECK-DAG: %[[VALUE:.+]] = arith.constant 123
  %value = arith.constant 123 : i32
  // CHECK: util.buffer.store %[[VALUE]], %[[BUFFER]][%[[OFFSET]] for {{.+}}] : i32 -> !util.buffer{%[[BUFFER_SIZE]]}
  stream.resource.store %value, %resource[%offset] : i32 -> !stream.resource<staging>{%resource_size}
  util.return
}

// -----

// CHECK-LABEL: @resourceSubview
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer, %[[BUFFER_SIZE:.+]]: index)
util.func public @resourceSubview(%resource: !stream.resource<transient>, %resource_size: index) -> !stream.resource<transient> {
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  // CHECK: %[[SUBSPAN:.+]] = util.buffer.subspan %[[BUFFER]][%c128] : !util.buffer{%[[BUFFER_SIZE]]} -> !util.buffer{%c256}
  %0 = stream.resource.subview %resource[%c128] : !stream.resource<transient>{%resource_size} -> !stream.resource<transient>{%c256}
  // CHECK: return %[[SUBSPAN]]
  util.return %0 : !stream.resource<transient>
}
