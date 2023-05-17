// RUN: iree-opt --split-input-file --iree-stream-refine-usage %s | FileCheck %s

// Tests that the refinement of a caller propagates into its callees.
// Here because %result is returned from the caller it becomes external, and
// because callee operates in-place it must also be external, and then the splat
// passed in must be external.

// CHECK-LABEL: @propagateFuncCallee
// CHECK-SAME: (%[[ARG:.+]]: !stream.resource<external>, %[[SIZE:.+]]: index) -> !stream.resource<external>
func.func private @propagateFuncCallee(%arg: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: stream.async.fill {{.+}} !stream.resource<external>
  %fill = stream.async.fill %c123_i32, %arg[%c0 to %c128 for %c128] : i32 -> %0 as !stream.resource<*>{%size}
  // CHECK: return {{.+}} : !stream.resource<external>
  return %fill : !stream.resource<*>
}
// CHECK: @propagateFuncCaller
// CHECK-SAME: -> !stream.resource<external>
func.func @propagateFuncCaller(%size: index) -> !stream.resource<*> {
  %c123_i32 = arith.constant 123 : i32
  // CHECK: stream.async.splat {{.+}} -> !stream.resource<external>
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: call @propagateFuncCallee({{.+}}) : (!stream.resource<external>, index) -> !stream.resource<external>
  %result = call @propagateFuncCallee(%splat, %size) : (!stream.resource<*>, index) -> !stream.resource<*>
  // CHECK: return {{.+}} : !stream.resource<external>
  return %result : !stream.resource<*>
}

// -----

// Tests that if a tied op (in this case export) is traversed during analysis
// and the type changes we don't explode.

// CHECK-LABEL: @transitionTypesAcrossTies
func.func @transitionTypesAcrossTies() -> !hal.buffer_view {
  %c4 = arith.constant 4 : index
  %c255_i32 = arith.constant 255 : i32
  // CHECK: %[[SPLAT:.+]] = stream.async.splat {{.+}} -> !stream.resource<external>
  %0 = stream.async.splat %c255_i32 : i32 -> !stream.resource<*>{%c4}
  // CHECK-NOT: stream.async.transfer
  %1 = stream.async.transfer %0 : !stream.resource<*>{%c4} -> !stream.resource<external>{%c4}
  // CHECK: stream.tensor.export %[[SPLAT]] : tensor<f32> in !stream.resource<external>{%c4} -> !hal.buffer_view
  %2 = stream.tensor.export %1 : tensor<f32> in !stream.resource<external>{%c4} -> !hal.buffer_view
  return %2 : !hal.buffer_view
}

// -----

// Tests that resource usage and type updates are propagated around the CFG.
// This starts with one result being * and the other pinned to external,
// demonstrating how we can hint usage when required and that the refinement
// will integrate the initial knowledge into the solver. Since all results are
// created within the function we know that if the splats that initially define
// them have the proper type then the propagation has reached through the blocks
// (and select).

// CHECK-LABEL: @propagateBlocks
// CHECK-SAME: (%[[COND:.+]]: i1, {{.+}}) -> (!stream.resource<transient>, !stream.resource<external>)
func.func private @propagateBlocks(%cond: i1, %size: index) -> (!stream.resource<*>, !stream.resource<external>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  // CHECK: %[[SPLAT0:.+]] = stream.async.splat %c123_i32 {{.+}} -> !stream.resource<transient>
  %splat0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[SPLAT1:.+]] = stream.async.splat %c456_i32 {{.+}} -> !stream.resource<external>
  %splat1 = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: cf.br ^bb1(%[[SPLAT0]], %[[SPLAT1]]
  cf.br ^bb1(%splat0, %splat1 : !stream.resource<*>, !stream.resource<*>)
// CHECK: ^bb1(%[[BB1_ARG0:.+]]: !stream.resource<transient>, %[[BB1_ARG1:.+]]: !stream.resource<external>)
^bb1(%bb1_0: !stream.resource<*>, %bb1_1: !stream.resource<*>):
  // CHECK: %[[CLONE0:.+]] = stream.async.clone %[[BB1_ARG0]] {{.+}} !stream.resource<transient>
  %clone0 = stream.async.clone %bb1_0 : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[FILL0:.+]] = stream.async.fill %c123_i32, %[[CLONE0]]{{.+}} !stream.resource<transient>
  %fill0 = stream.async.fill %c123_i32, %clone0[%c0 to %c128 for %c128] : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[CLONE1:.+]] = stream.async.clone %[[BB1_ARG1]] {{.+}} !stream.resource<external>
  %clone1 = stream.async.clone %bb1_1 : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[FILL1:.+]] = stream.async.fill %c456_i32, %[[CLONE1]]{{.+}} !stream.resource<external>
  %fill1 = stream.async.fill %c456_i32, %clone1[%c0 to %c128 for %c128] : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[SELECT:.+]] = arith.select %[[COND]], %[[SPLAT1]], %[[FILL1]] : !stream.resource<external>
  %bb1_1_new = arith.select %cond, %splat1, %fill1 : !stream.resource<*>
  // CHECK: cf.cond_br %[[COND]], ^bb1(%[[FILL0]], %[[SELECT]]
  // CHECK-SAME:               ^bb2(%[[FILL0]], %[[SELECT]]
  cf.cond_br %cond, ^bb1(%fill0, %bb1_1_new : !stream.resource<*>, !stream.resource<*>),
                 ^bb2(%fill0, %bb1_1_new : !stream.resource<*>, !stream.resource<*>)
// CHECK: ^bb2(%[[BB2_ARG0:.+]]: !stream.resource<transient>, %[[BB2_ARG1:.+]]: !stream.resource<external>)
^bb2(%bb2_0: !stream.resource<*>, %bb2_1: !stream.resource<*>):
  // CHECK-NOT: stream.async.transfer
  %external_transfer = stream.async.transfer %bb2_1 : !stream.resource<*>{%size} -> !stream.resource<external>{%size}
  // CHECK: return %[[BB2_ARG0]], %[[BB2_ARG1]] : !stream.resource<transient>, !stream.resource<external>
  return %bb2_0, %external_transfer : !stream.resource<*>, !stream.resource<external>
}

// -----

// Tests conflict resolution.
// External is wider than transient so we expect the transient to be widened.

// CHECK-LABEL: @conflictResolution
// CHECK-SAME: (%[[COND:.+]]: i1, %[[ARG0:.+]]: !stream.resource<transient>, %[[ARG1:.+]]: !stream.resource<external>, %[[SIZE:.+]]: index)
// CHECK-SAME: -> !stream.resource<external>
func.func @conflictResolution(%cond: i1, %arg0: !stream.resource<transient>, %arg1: !stream.resource<external>, %size: index) -> !stream.resource<*> {
  // CHECK: %[[ARG0_EXT:.+]] = stream.async.transfer %[[ARG0]]
  %arg0_any = stream.async.transfer %arg0 : !stream.resource<transient>{%size} -> !stream.resource<*>{%size}
  // CHECK-NOT: stream.async.transfer %[[ARG1]]
  %arg1_any = stream.async.transfer %arg1 : !stream.resource<external>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[RET:.+]] = arith.select %[[COND]], %[[ARG0_EXT]], %[[ARG1]] : !stream.resource<external>
  %0 = arith.select %cond, %arg0_any, %arg1_any : !stream.resource<*>
  // CHECK: return %[[RET]] : !stream.resource<external>
  return %0 : !stream.resource<*>
}

// -----

// Tests invalid transfer conflict resolution.
// Constants cannot be mutated even though it is tied. This survives after
// copy-on-write materialization because of the transfer and we need to preserve
// it such that the copy is performed as epxected.

// CHECK-LABEL: @transferResolution
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<constant>, %[[SIZE:.+]]: index)
// CHECK-SAME: -> !stream.resource<external>
func.func @transferResolution(%arg0: !stream.resource<constant>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[ARG0_EXT:.+]] = stream.async.transfer %[[ARG0]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  %arg0_any = stream.async.transfer %arg0 : !stream.resource<constant>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[RET0:.+]] = stream.async.dispatch @ex::@dispatch[%c1, %c1, %c1](%[[ARG0_EXT]][%c0 to %[[SIZE]] for %[[SIZE]]]) : (!stream.resource<external>{%[[SIZE]]}) -> %[[ARG0_EXT]]{%[[SIZE]]}
  %ret0_any = stream.async.dispatch @ex::@dispatch[%c1, %c1, %c1](%arg0_any[%c0 to %size for %size]) : (!stream.resource<*>{%size}) -> %arg0_any{%size}
  // return %[[RET0]] : !stream.resource<external>
  return %ret0_any : !stream.resource<*>
}

// -----

// Tests that multiple transfers are elided during transfer materialization.

// CHECK-LABEL: @transferElision
// CHECK-SAME: (%[[SIZE:.+]]: index) -> !stream.resource<external>
func.func @transferElision(%size: index) -> !stream.resource<external> {
  // CHECK: %[[ALLOCA:.+]] = stream.async.alloca
  %alloca = stream.async.alloca : !stream.resource<constant>{%size}
  %transfer_any = stream.async.transfer %alloca : !stream.resource<constant>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[TRANSFER_EXTERNAL:.+]] = stream.async.transfer %[[ALLOCA]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  %transfer_external = stream.async.transfer %transfer_any : !stream.resource<*>{%size} -> !stream.resource<external>{%size}
  // CHECK: return %[[TRANSFER_EXTERNAL]]
  return %transfer_external : !stream.resource<external>
}

// -----

// Tests that global usage propagates through loads/stores.

util.global private mutable @variable : !stream.resource<variable>
util.global private mutable @variable__size : index

// CHECK-LABEL: @globalLoad()
// CHECK-SAME: -> !stream.resource<variable>
func.func private @globalLoad() -> !stream.resource<*> {
  // CHECK: %[[VALUE:.+]] = util.global.load @variable : !stream.resource<variable>
  %value = util.global.load @variable : !stream.resource<variable>
  %size = util.global.load @variable__size : index
  // CHECK-NOT: stream.async.transfer
  %0 = stream.async.transfer %value : !stream.resource<variable>{%size} -> !stream.resource<*>{%size}
  // CHECK: return %[[VALUE]]
  return %0 : !stream.resource<*>
}

// CHECK-LABEL: @globalStore
// CHECK-SAME: (%[[VALUE:.+]]: !stream.resource<variable>, %[[SIZE:.+]]: index)
func.func private @globalStore(%value: !stream.resource<*>, %size: index) {
  // CHECK-NOT: stream.async.transfer
  %0 = stream.async.transfer %value : !stream.resource<*>{%size} -> !stream.resource<variable>{%size}
  // CHECK: util.global.store %[[VALUE]], @variable : !stream.resource<variable>
  util.global.store %0, @variable : !stream.resource<variable>
  util.global.store %size, @variable__size : index
  return
}

// -----

// Tests that explicit resource allocations are refined.

// CHECK-LABEL: @explicitAlloc
func.func @explicitAlloc() -> !hal.buffer_view {
  %c0 = arith.constant 0 : index
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc : !stream.resource<external>{%c0}
  %0 = stream.resource.alloc : !stream.resource<*>{%c0}
  // CHECK-NOT: stream.async.transfer
  %1 = stream.async.transfer %0 : !stream.resource<*>{%c0} -> !stream.resource<external>{%c0}
  // CHECK: stream.tensor.export %[[ALLOC]] : tensor<f32> in !stream.resource<external>{%c0} -> !hal.buffer_view
  %2 = stream.tensor.export %1 : tensor<f32> in !stream.resource<external>{%c0} -> !hal.buffer_view
  return %2 : !hal.buffer_view
}

// -----

// Tests that async allocations that escape are turned into non-transient allocs.

// CHECK-LABEL: @escapingAlloca
func.func @escapingAlloca() -> !hal.buffer_view {
  %c123 = arith.constant 123 : index
  // CHECK: %[[ALLOCA:.+]] = stream.async.alloca : !stream.resource<external>{%c123}
  %0 = stream.async.alloca : !stream.resource<*>{%c123}
  // CHECK-NOT: stream.async.transfer
  %1 = stream.async.transfer %0 : !stream.resource<*>{%c123} -> !stream.resource<external>{%c123}
  // CHECK: stream.tensor.export %[[ALLOCA]] : tensor<f32> in !stream.resource<external>{%c123} -> !hal.buffer_view
  %2 = stream.tensor.export %1 : tensor<f32> in !stream.resource<external>{%c123} -> !hal.buffer_view
  return %2 : !hal.buffer_view
}
