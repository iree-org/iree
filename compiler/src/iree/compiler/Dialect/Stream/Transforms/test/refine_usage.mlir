// RUN: iree-opt --split-input-file --iree-stream-refine-usage %s | FileCheck %s

// Tests that the refinement of a caller propagates into its callees.
// Here because %result is returned from the caller it becomes external, and
// because callee operates in-place it must also be external, and then the splat
// passed in must be external.

// CHECK-LABEL: @propagateFuncCallee
// CHECK-SAME: (%[[ARG:.+]]: !stream.resource<external>, %[[SIZE:.+]]: index) -> !stream.resource<external>
util.func private @propagateFuncCallee(%arg: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: stream.async.fill {{.+}} !stream.resource<external>
  %fill = stream.async.fill %c123_i32, %arg[%c0 to %c128 for %c128] : i32 -> %0 as !stream.resource<*>{%size}
  // CHECK: util.return {{.+}} : !stream.resource<external>
  util.return %fill : !stream.resource<*>
}
// CHECK: @propagateFuncCaller
// CHECK-SAME: -> !stream.resource<external>
util.func public @propagateFuncCaller(%size: index) -> !stream.resource<*> {
  %c123_i32 = arith.constant 123 : i32
  // CHECK: stream.async.splat {{.+}} -> !stream.resource<external>
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: call @propagateFuncCallee({{.+}}) : (!stream.resource<external>, index) -> !stream.resource<external>
  %result = util.call @propagateFuncCallee(%splat, %size) : (!stream.resource<*>, index) -> !stream.resource<*>
  // CHECK: util.return {{.+}} : !stream.resource<external>
  util.return %result : !stream.resource<*>
}

// -----

// Tests that if a tied op (in this case export) is traversed during analysis
// and the type changes we don't explode.

// CHECK-LABEL: @transitionTypesAcrossTies
util.func public @transitionTypesAcrossTies() -> !hal.buffer_view {
  %c4 = arith.constant 4 : index
  %c255_i32 = arith.constant 255 : i32
  // CHECK: %[[SPLAT:.+]] = stream.async.splat {{.+}} -> !stream.resource<external>
  %0 = stream.async.splat %c255_i32 : i32 -> !stream.resource<*>{%c4}
  // CHECK-NOT: stream.async.transfer
  %1 = stream.async.transfer %0 : !stream.resource<*>{%c4} -> !stream.resource<external>{%c4}
  // CHECK: stream.tensor.export %[[SPLAT]] : tensor<f32> in !stream.resource<external>{%c4} -> !hal.buffer_view
  %2 = stream.tensor.export %1 : tensor<f32> in !stream.resource<external>{%c4} -> !hal.buffer_view
  util.return %2 : !hal.buffer_view
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
util.func private @propagateBlocks(%cond: i1, %size: index) -> (!stream.resource<*>, !stream.resource<external>) {
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
  // CHECK: util.return %[[BB2_ARG0]], %[[BB2_ARG1]] : !stream.resource<transient>, !stream.resource<external>
  util.return %bb2_0, %external_transfer : !stream.resource<*>, !stream.resource<external>
}

// -----

// Tests conflict resolution.
// External is wider than transient so we expect the transient to be widened.

// CHECK-LABEL: @conflictResolution
// CHECK-SAME: (%[[COND:.+]]: i1, %[[ARG0:.+]]: !stream.resource<transient>, %[[ARG1:.+]]: !stream.resource<external>, %[[SIZE:.+]]: index)
// CHECK-SAME: -> !stream.resource<external>
util.func public @conflictResolution(%cond: i1, %arg0: !stream.resource<transient>, %arg1: !stream.resource<external>, %size: index) -> !stream.resource<*> {
  // CHECK: %[[ARG0_EXT:.+]] = stream.async.transfer %[[ARG0]]
  %arg0_any = stream.async.transfer %arg0 : !stream.resource<transient>{%size} -> !stream.resource<*>{%size}
  // CHECK-NOT: stream.async.transfer %[[ARG1]]
  %arg1_any = stream.async.transfer %arg1 : !stream.resource<external>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[RET:.+]] = arith.select %[[COND]], %[[ARG0_EXT]], %[[ARG1]] : !stream.resource<external>
  %0 = arith.select %cond, %arg0_any, %arg1_any : !stream.resource<*>
  // CHECK: util.return %[[RET]] : !stream.resource<external>
  util.return %0 : !stream.resource<*>
}

// -----

// Tests invalid transfer conflict resolution.
// Constants cannot be mutated even though it is tied. This survives after
// copy-on-write materialization because of the transfer and we need to preserve
// it such that the copy is performed as epxected.

// CHECK-LABEL: @transferResolution
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<constant>, %[[SIZE:.+]]: index)
// CHECK-SAME: -> !stream.resource<external>
util.func public @transferResolution(%arg0: !stream.resource<constant>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[ARG0_EXT:.+]] = stream.async.transfer %[[ARG0]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  %arg0_any = stream.async.transfer %arg0 : !stream.resource<constant>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[RET0:.+]] = stream.async.dispatch @ex::@dispatch[%c1, %c1, %c1](%[[ARG0_EXT]][%c0 to %[[SIZE]] for %[[SIZE]]]) : (!stream.resource<external>{%[[SIZE]]}) -> %[[ARG0_EXT]]{%[[SIZE]]}
  %ret0_any = stream.async.dispatch @ex::@dispatch[%c1, %c1, %c1](%arg0_any[%c0 to %size for %size]) : (!stream.resource<*>{%size}) -> %arg0_any{%size}
  // return %[[RET0]] : !stream.resource<external>
  util.return %ret0_any : !stream.resource<*>
}

// -----

// Tests that multiple transfers are elided during transfer materialization.

// CHECK-LABEL: @transferElision
// CHECK-SAME: (%[[SIZE:.+]]: index) -> !stream.resource<external>
util.func public @transferElision(%size: index) -> !stream.resource<external> {
  // CHECK: %[[ALLOCA:.+]] = stream.async.alloca
  %alloca = stream.async.alloca : !stream.resource<constant>{%size}
  %transfer_any = stream.async.transfer %alloca : !stream.resource<constant>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[TRANSFER_EXTERNAL:.+]] = stream.async.transfer %[[ALLOCA]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  %transfer_external = stream.async.transfer %transfer_any : !stream.resource<*>{%size} -> !stream.resource<external>{%size}
  // CHECK: util.return %[[TRANSFER_EXTERNAL]]
  util.return %transfer_external : !stream.resource<external>
}

// -----

// Tests that global usage propagates through loads/stores.

util.global private mutable @variable : !stream.resource<variable>
util.global private mutable @variable__size : index

// CHECK-LABEL: @globalLoad()
// CHECK-SAME: -> !stream.resource<variable>
util.func private @globalLoad() -> !stream.resource<*> {
  // CHECK: %[[VALUE:.+]] = util.global.load @variable : !stream.resource<variable>
  %value = util.global.load @variable : !stream.resource<variable>
  %size = util.global.load @variable__size : index
  // CHECK-NOT: stream.async.transfer
  %0 = stream.async.transfer %value : !stream.resource<variable>{%size} -> !stream.resource<*>{%size}
  // CHECK: util.return %[[VALUE]]
  util.return %0 : !stream.resource<*>
}

// CHECK-LABEL: @globalStore
// CHECK-SAME: (%[[VALUE:.+]]: !stream.resource<variable>, %[[SIZE:.+]]: index)
util.func private @globalStore(%value: !stream.resource<*>, %size: index) {
  // CHECK-NOT: stream.async.transfer
  %0 = stream.async.transfer %value : !stream.resource<*>{%size} -> !stream.resource<variable>{%size}
  // CHECK: util.global.store %[[VALUE]], @variable : !stream.resource<variable>
  util.global.store %0, @variable : !stream.resource<variable>
  util.global.store %size, @variable__size : index
  util.return
}

// -----

// Tests that explicit resource allocations are refined.

// CHECK-LABEL: @explicitAlloc
util.func public @explicitAlloc() -> !hal.buffer_view {
  %c0 = arith.constant 0 : index
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc : !stream.resource<external>{%c0}
  %0 = stream.resource.alloc : !stream.resource<*>{%c0}
  // CHECK-NOT: stream.async.transfer
  %1 = stream.async.transfer %0 : !stream.resource<*>{%c0} -> !stream.resource<external>{%c0}
  // CHECK: stream.tensor.export %[[ALLOC]] : tensor<f32> in !stream.resource<external>{%c0} -> !hal.buffer_view
  %2 = stream.tensor.export %1 : tensor<f32> in !stream.resource<external>{%c0} -> !hal.buffer_view
  util.return %2 : !hal.buffer_view
}

// -----

// Tests that async allocations that escape are turned into non-transient allocs.

// CHECK-LABEL: @escapingAlloca
util.func public @escapingAlloca() -> !hal.buffer_view {
  %c123 = arith.constant 123 : index
  // CHECK: %[[ALLOCA:.+]] = stream.async.alloca : !stream.resource<external>{%c123}
  %0 = stream.async.alloca : !stream.resource<*>{%c123}
  // CHECK-NOT: stream.async.transfer
  %1 = stream.async.transfer %0 : !stream.resource<*>{%c123} -> !stream.resource<external>{%c123}
  // CHECK: stream.tensor.export %[[ALLOCA]] : tensor<f32> in !stream.resource<external>{%c123} -> !hal.buffer_view
  %2 = stream.tensor.export %1 : tensor<f32> in !stream.resource<external>{%c123} -> !hal.buffer_view
  util.return %2 : !hal.buffer_view
}

// -----

// CHECK-LABEL: @testIf
util.func public @testIf(%arg0: i1, %arg1: !stream.resource<*>, %arg2: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[IF:.+]] = scf.if
  // CHECK-SAME: !stream.resource<external>
  %if = scf.if %arg0 -> (!stream.resource<*>) {
    // CHECK: %[[DISPATCH:.+]] = stream.async.dispatch
    // CHECK-SAME: !stream.resource<external>
    // CHECK-SAME: !stream.resource<external>
    // CHECK-SAME: -> !stream.resource<external>
    %disp = stream.async.dispatch @disp(%arg1[%c0 to %c4 for %c4], %arg2[%c0 to %c4 for %c4]) : (!stream.resource<*>{%c4}, !stream.resource<*>{%c4}) -> !stream.resource<*>{%c4}
    // CHECK: scf.yield
    // CHECK-SAME: !stream.resource<external>
    scf.yield %disp : !stream.resource<*>
  } else {
    // CHECK: scf.yield
    // CHECK-SAME: !stream.resource<external>
    scf.yield %arg1 : !stream.resource<*>
  }
  util.return %if : !stream.resource<*>
}

// -----

// CHECK: @testWhile
util.func public @testWhile(%arg0: i32, %arg1: !stream.resource<*>) -> (i32, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : i32
  // CHECK: scf.while
  // CHECK-SAME: (i32, !stream.resource<external>)
  // CHECK-SAME: (i32, !stream.resource<external>)
  %while:2 = scf.while (%arg2 = %arg0, %arg3 = %arg1) : (i32, !stream.resource<*>) -> (i32, !stream.resource<*>) {
    %cmp = arith.cmpi slt, %arg2, %c10 : i32
    // CHECK: scf.condition
    // CHECK-SAME: !stream.resource<external>
    scf.condition(%cmp) %arg2, %arg3 : i32, !stream.resource<*>
  } do {
  ^bb0(%arg2: i32, %arg3: !stream.resource<*>):
    %add = arith.addi %arg2, %c1 : i32
    %disp = stream.async.dispatch @disp(%arg3[%c0 to %c4 for %c4], %arg1[%c0 to %c4 for %c4]) : (!stream.resource<*>{%c4}, !stream.resource<*>{%c4}) -> !stream.resource<*>{%c4}
    // CHECK: scf.yield
    // CHECK-SAME: !stream.resource<external>
    scf.yield %add, %disp : i32, !stream.resource<*>
  }
  // CHECK: util.return %[[IF]]#0, %[[IF]]#1 : i32, !stream.resource<external>
  util.return %while#0, %while#1 : i32, !stream.resource<*>
}

// -----

// CHECK-LABEL: @testWhileRecurse
// CHECK-SAME: %[[ARG0:.+]]: !stream.resource<external>
// CHECK-SAME: -> !stream.resource<external>
util.func public @testWhileRecurse(%arg0 : !stream.resource<*>) -> !stream.resource<external> {
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  // CHECK: %[[SIZE:.+]] = stream.resource.size %[[ARG0]] : !stream.resource<external>
  %size = stream.resource.size %arg0 : !stream.resource<*>

  // CHECK: %[[WHILE:.+]]:2 = scf.while (%[[ARG1:.+]] = %[[ARG0]], %[[ARG2:.+]] = %[[SIZE]]) : (!stream.resource<external>, index) -> (!stream.resource<external>, index)
  %while:2 = scf.while (%arg1 = %arg0, %arg2 = %size) : (!stream.resource<*>, index) -> (!stream.resource<*>, index) {
    // CHECK: %[[DISPATCH:.+]] = stream.async.dispatch @dispatch(%[[ARG1]][%[[C0]] to %[[ARG2]] for %[[ARG2]]]) : (!stream.resource<external>{%[[ARG2]]}) -> !stream.resource<transient>{%[[C1]]}
    %dispatch = stream.async.dispatch @dispatch(%arg1[%c0 to %arg2 for %arg2]) : (!stream.resource<*>{%arg2}) -> !stream.resource<*>{%c1}

    // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[DISPATCH]] : !stream.resource<transient>{%[[C1]]} -> !stream.resource<staging>{%[[C1]]}
    %transfer = stream.async.transfer %dispatch : !stream.resource<*>{%c1} -> !stream.resource<staging>{%c1}

    // CHECK: %[[LOAD:.+]] = stream.async.load %[[TRANSFER]][%[[C0]]] : !stream.resource<staging>{%[[C1]]} -> i1
    %load = stream.async.load %transfer[%c0] : !stream.resource<staging>{%c1} -> i1

    // CHECK: scf.condition(%[[LOAD]]) %[[ARG1]], %[[ARG2]] : !stream.resource<external>, index
    scf.condition(%load) %arg1, %arg2 : !stream.resource<*>, index
  } do {
  // CHECK: ^bb0(%[[ARG1:.+]]: !stream.resource<external>, %[[ARG2:.+]]: index):
  ^bb0(%arg1: !stream.resource<*>, %arg2: index):
    // CHECK: %[[DISPATCH:.+]] = stream.async.dispatch @dispatch(%[[ARG1]][%[[C0]] to %[[ARG2]] for %[[ARG2]]]) : (!stream.resource<external>{%[[ARG2]]}) -> !stream.resource<external>{%[[C4]]}
    %dispatch = stream.async.dispatch @dispatch(%arg1[%c0 to %arg2 for %arg2]) : (!stream.resource<*>{%arg2}) -> !stream.resource<*>{%c4}

    // CHECK: scf.yield %[[DISPATCH]], %[[C4]] : !stream.resource<external>, index
    scf.yield %dispatch, %c4 : !stream.resource<*>, index
  }
  %transfer = stream.async.transfer %while#0 : !stream.resource<*>{%while#1} -> !stream.resource<external>{%while#1}

  // CHECK: util.return %[[WHILE]]#0
  util.return %transfer : !stream.resource<external>
}

// -----

// CHECK-LABEL: @testForOp
// CHECK-SAME: %[[ARG0:.+]]: index
// CHECK-SAME: %[[ARG1:.+]]: !stream.resource<external>
util.func public @testForOp(%arg0 : index, %arg1 : !stream.resource<*>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[C1:.+]] = arith.constant 1 : index
  // CHECK: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: %[[DISP0:.+]] = stream.async.dispatch @dispatch0(%arg1[%[[C0]] to %[[ARG0]] for %[[ARG0]]]) : (!stream.resource<external>{%[[C4]]}) -> !stream.resource<transient>{%[[C4]]}
  %dispatch6 = stream.async.dispatch @dispatch0(%arg1[%c0 to %arg0 for %arg0]) : (!stream.resource<*>{%c4}) -> !stream.resource<*>{%c4}

  // CHECK: %[[FOR:.+]] = scf.for %[[ARG2:.+]] = %[[C0]] to %[[ARG0]] step %[[C1]] iter_args(%[[ARG3:.+]] = %[[DISP0]]) -> (!stream.resource<transient>) {
  %for = scf.for %i = %c0 to %arg0 step %c1 iter_args(%arg3 = %dispatch6) -> (!stream.resource<*>) {
    // CHECK:   %[[DISP1:.+]] = stream.async.dispatch @dispatch1(%[[ARG3]][%[[C0]] to %[[ARG0]] for %[[ARG0]]]) : (!stream.resource<transient>{%[[C4]]}) -> !stream.resource<transient>{%[[C4]]}
    // CHECK:   %[[DISP2:.+]] = stream.async.dispatch @dispatch2(%[[DISP1]][%[[C0]] to %[[ARG0]] for %[[ARG0]]]) : (!stream.resource<transient>{%[[C4]]}) -> !stream.resource<transient>{%[[C4]]}
    // CHECK:   %[[DISP3:.+]] = stream.async.dispatch @dispatch3(%[[DISP2]][%[[C0]] to %[[ARG0]] for %[[ARG0]]]) : (!stream.resource<transient>{%[[C4]]}) -> !stream.resource<transient>{%[[C4]]}
    %dispatch1 = stream.async.dispatch @dispatch1(%arg3[%c0 to %arg0 for %arg0]) : (!stream.resource<*>{%c4}) -> !stream.resource<*>{%c4}
    %dispatch2 = stream.async.dispatch @dispatch2(%dispatch1[%c0 to %arg0 for %arg0]) : (!stream.resource<*>{%c4}) -> !stream.resource<*>{%c4}
    %dispatch3 = stream.async.dispatch @dispatch3(%dispatch2[%c0 to %arg0 for %arg0]) : (!stream.resource<*>{%c4}) -> !stream.resource<*>{%c4}
    // CHECK:   scf.yield %[[DISP3]] : !stream.resource<transient>
    scf.yield %dispatch3 : !stream.resource<*>
  }

  // CHECK: %[[DISP4:.+]] = stream.async.dispatch @dispatch4(%[[FOR]][%[[C0]] to %[[ARG0]] for %[[ARG0]]]) : (!stream.resource<transient>{%[[C4]]}) -> !stream.resource<external>{%[[C4]]}
  %dispatch5 = stream.async.dispatch @dispatch4(%for[%c0 to %arg0 for %arg0]) : (!stream.resource<*>{%c4}) -> !stream.resource<*>{%c4}
  %transfer = stream.async.transfer %dispatch5 : !stream.resource<*>{%arg0} -> !stream.resource<external>{%arg0}

  // CHECK: util.return %[[DISP4]] : !stream.resource<external>
  util.return %transfer : !stream.resource<external>
}
