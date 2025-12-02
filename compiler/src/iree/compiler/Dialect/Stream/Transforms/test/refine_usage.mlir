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
// and the type changes we don't explode. The transfer from * to external is
// preserved by RefineUsagePass and will be elided by ElideAsyncCopiesPass since
// it's a same-type transfer (external->external after refinement).

// CHECK-LABEL: @transitionTypesAcrossTies
util.func public @transitionTypesAcrossTies() -> !util.buffer {
  %c4 = arith.constant 4 : index
  %c255_i32 = arith.constant 255 : i32
  // CHECK: %[[SPLAT:.+]] = stream.async.splat {{.+}} -> !stream.resource<external>
  %0 = stream.async.splat %c255_i32 : i32 -> !stream.resource<*>{%c4}
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[SPLAT]] : !stream.resource<external>{%c4} -> !stream.resource<external>{%c4}
  %1 = stream.async.transfer %0 : !stream.resource<*>{%c4} -> !stream.resource<external>{%c4}
  // CHECK: stream.tensor.export %[[TRANSFER]] : tensor<f32> in !stream.resource<external>{%c4} -> !util.buffer
  %2 = stream.tensor.export %1 : tensor<f32> in !stream.resource<external>{%c4} -> !util.buffer
  util.return %2 : !util.buffer
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
  // CHECK-SAME:               ^bb2
  cf.cond_br %cond, ^bb1(%fill0, %bb1_1_new : !stream.resource<*>, !stream.resource<*>),
                 ^bb2(%fill0, %bb1_1_new : !stream.resource<*>, !stream.resource<*>)
// CHECK: ^bb2
^bb2(%bb2_0: !stream.resource<*>, %bb2_1: !stream.resource<*>):
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[SELECT]] : !stream.resource<external>{{.+}} -> !stream.resource<external>
  %external_transfer = stream.async.transfer %bb2_1 : !stream.resource<*>{%size} -> !stream.resource<external>{%size}
  // CHECK: util.return %[[FILL0]], %[[TRANSFER]] : !stream.resource<transient>, !stream.resource<external>
  util.return %bb2_0, %external_transfer : !stream.resource<*>, !stream.resource<external>
}

// -----

// Tests conflict resolution.
// External is wider than transient so we expect the transient to be widened.

// CHECK-LABEL: @conflictResolution
// CHECK-SAME: (%[[COND:.+]]: i1, %[[ARG0:.+]]: !stream.resource<transient>, %[[ARG1:.+]]: !stream.resource<external>, %[[SIZE:.+]]: index)
// CHECK-SAME: -> !stream.resource<external>
util.func public @conflictResolution(%cond: i1, %arg0: !stream.resource<transient>, %arg1: !stream.resource<external>, %size: index) -> !stream.resource<*> {
  // CHECK: %[[ARG0_EXT:.+]] = stream.async.transfer %[[ARG0]] : !stream.resource<transient>{%[[SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  %arg0_any = stream.async.transfer %arg0 : !stream.resource<transient>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[ARG1_EXT:.+]] = stream.async.transfer %[[ARG1]] : !stream.resource<external>{%[[SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  %arg1_any = stream.async.transfer %arg1 : !stream.resource<external>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[RET:.+]] = arith.select %[[COND]], %[[ARG0_EXT]], %[[ARG1_EXT]] : !stream.resource<external>
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

// Tests that transfer chains are preserved during refinement. The chain
// constant->*->external becomes constant->transient->external after refinement.
// Note: A redundant external->external transfer may appear but will be
// eliminated by ElideAsyncCopiesPass.

// CHECK-LABEL: @transferElision
// CHECK-SAME: (%[[SIZE:.+]]: index) -> !stream.resource<external>
util.func public @transferElision(%size: index) -> !stream.resource<external> {
  // CHECK: %[[ALLOCA:.+]] = stream.async.alloca : !stream.resource<constant>{%[[SIZE]]}
  %alloca = stream.async.alloca : !stream.resource<constant>{%size}
  // CHECK: %[[TRANSFER_TRANSIENT:.+]] = stream.async.transfer %[[ALLOCA]] : !stream.resource<constant>{%[[SIZE]]} -> !stream.resource<transient>{%[[SIZE]]}
  %transfer_any = stream.async.transfer %alloca : !stream.resource<constant>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[TRANSFER_EXTERNAL:.+]] = stream.async.transfer %[[TRANSFER_TRANSIENT]] : !stream.resource<transient>{%[[SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  %transfer_external = stream.async.transfer %transfer_any : !stream.resource<*>{%size} -> !stream.resource<external>{%size}
  // A redundant transfer may be inserted here but will be eliminated later.
  // CHECK: %[[TRANSFER_REDUNDANT:.+]] = stream.async.transfer %[[TRANSFER_EXTERNAL]] : !stream.resource<external>{%[[SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
  // CHECK: util.return %[[TRANSFER_REDUNDANT]] : !stream.resource<external>
  util.return %transfer_external : !stream.resource<external>
}

// -----

// Tests that global usage propagates through loads/stores. Same-type transfers
// (variable->variable) are preserved during refinement and will be elided by
// ElideAsyncCopiesPass.

util.global private mutable @variable : !stream.resource<variable>
util.global private mutable @variable__size : index

// CHECK-LABEL: @globalLoad()
// CHECK-SAME: -> !stream.resource<variable>
util.func private @globalLoad() -> !stream.resource<*> {
  // CHECK: %[[VALUE:.+]] = util.global.load @variable : !stream.resource<variable>
  %value = util.global.load @variable : !stream.resource<variable>
  %size = util.global.load @variable__size : index
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[VALUE]] : !stream.resource<variable>{%{{.+}}} -> !stream.resource<variable>{%{{.+}}}
  %0 = stream.async.transfer %value : !stream.resource<variable>{%size} -> !stream.resource<*>{%size}
  // CHECK: util.return %[[TRANSFER]] : !stream.resource<variable>
  util.return %0 : !stream.resource<*>
}

// CHECK-LABEL: @globalStore
// CHECK-SAME: (%[[VALUE:.+]]: !stream.resource<variable>, %[[SIZE:.+]]: index)
util.func private @globalStore(%value: !stream.resource<*>, %size: index) {
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[VALUE]] : !stream.resource<variable>{%[[SIZE]]} -> !stream.resource<variable>{%[[SIZE]]}
  %0 = stream.async.transfer %value : !stream.resource<*>{%size} -> !stream.resource<variable>{%size}
  // CHECK: util.global.store %[[TRANSFER]], @variable : !stream.resource<variable>
  util.global.store %0, @variable : !stream.resource<variable>
  util.global.store %size, @variable__size : index
  util.return
}

// -----

// Tests that explicit resource allocations are refined. Same-type transfer
// (external->external) is preserved during refinement.

// CHECK-LABEL: @explicitAlloc
util.func public @explicitAlloc() -> !util.buffer {
  %c0 = arith.constant 0 : index
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc : !stream.resource<external>{%c0}
  %0 = stream.resource.alloc : !stream.resource<*>{%c0}
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[ALLOC]] : !stream.resource<external>{%c0} -> !stream.resource<external>{%c0}
  %1 = stream.async.transfer %0 : !stream.resource<*>{%c0} -> !stream.resource<external>{%c0}
  // CHECK: stream.tensor.export %[[TRANSFER]] : tensor<f32> in !stream.resource<external>{%c0} -> !util.buffer
  %2 = stream.tensor.export %1 : tensor<f32> in !stream.resource<external>{%c0} -> !util.buffer
  util.return %2 : !util.buffer
}

// -----

// Tests that async allocations that escape are turned into non-transient allocs.
// Same-type transfer (external->external) is preserved during refinement.

// CHECK-LABEL: @escapingAlloca
util.func public @escapingAlloca() -> !util.buffer {
  %c123 = arith.constant 123 : index
  // CHECK: %[[ALLOCA:.+]] = stream.async.alloca : !stream.resource<external>{%c123}
  %0 = stream.async.alloca : !stream.resource<*>{%c123}
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[ALLOCA]] : !stream.resource<external>{%c123} -> !stream.resource<external>{%c123}
  %1 = stream.async.transfer %0 : !stream.resource<*>{%c123} -> !stream.resource<external>{%c123}
  // CHECK: stream.tensor.export %[[TRANSFER]] : tensor<f32> in !stream.resource<external>{%c123} -> !util.buffer
  %2 = stream.tensor.export %1 : tensor<f32> in !stream.resource<external>{%c123} -> !util.buffer
  util.return %2 : !util.buffer
}

// -----

// Tests scf.if with resources. Both branches must yield the same type, and
// function arguments are refined based on usage in both branches.

// CHECK-LABEL: @testIf
// CHECK-SAME: (%[[COND:.+]]: i1, %[[ARG1:.+]]: !stream.resource<external>, %[[ARG2:.+]]: !stream.resource<external>)
// CHECK-SAME: -> !stream.resource<external>
util.func public @testIf(%arg0: i1, %arg1: !stream.resource<*>, %arg2: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[IF:.+]] = scf.if %[[COND]] -> (!stream.resource<external>)
  %if = scf.if %arg0 -> (!stream.resource<*>) {
    // CHECK: %[[DISPATCH:.+]] = stream.async.dispatch @disp(%[[ARG1]][%c0 to %c4 for %c4], %[[ARG2]][%c0 to %c4 for %c4]) : (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}) -> !stream.resource<external>{%c4}
    %disp = stream.async.dispatch @disp(%arg1[%c0 to %c4 for %c4], %arg2[%c0 to %c4 for %c4]) : (!stream.resource<*>{%c4}, !stream.resource<*>{%c4}) -> !stream.resource<*>{%c4}
    // CHECK: scf.yield %[[DISPATCH]] : !stream.resource<external>
    scf.yield %disp : !stream.resource<*>
  } else {
    // CHECK: scf.yield %[[ARG1]] : !stream.resource<external>
    scf.yield %arg1 : !stream.resource<*>
  }
  // CHECK: util.return %[[IF]] : !stream.resource<external>
  util.return %if : !stream.resource<*>
}

// -----

// Tests scf.while with resources. Loop arguments are refined based on usage
// across both before and after regions.

// CHECK-LABEL: @testWhile
// CHECK-SAME: (%[[ARG0:.+]]: i32, %[[ARG1:.+]]: !stream.resource<external>)
// CHECK-SAME: -> (i32, !stream.resource<external>)
util.func public @testWhile(%arg0: i32, %arg1: !stream.resource<*>) -> (i32, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : i32
  // CHECK: %[[WHILE:.+]]:2 = scf.while (%[[ARG2:.+]] = %[[ARG0]], %[[ARG3:.+]] = %[[ARG1]]) : (i32, !stream.resource<external>) -> (i32, !stream.resource<external>)
  %while:2 = scf.while (%arg2 = %arg0, %arg3 = %arg1) : (i32, !stream.resource<*>) -> (i32, !stream.resource<*>) {
    %cmp = arith.cmpi slt, %arg2, %c10 : i32
    // CHECK: scf.condition(%{{.+}}) %[[ARG2]], %[[ARG3]] : i32, !stream.resource<external>
    scf.condition(%cmp) %arg2, %arg3 : i32, !stream.resource<*>
  } do {
  ^bb0(%arg2: i32, %arg3: !stream.resource<*>):
    %add = arith.addi %arg2, %c1 : i32
    // CHECK: %[[DISPATCH:.+]] = stream.async.dispatch @disp(%[[ARG3]][%c0 to %c4 for %c4], %[[ARG1]][%c0 to %c4 for %c4]) : (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}) -> !stream.resource<external>{%c4}
    %disp = stream.async.dispatch @disp(%arg3[%c0 to %c4 for %c4], %arg1[%c0 to %c4 for %c4]) : (!stream.resource<*>{%c4}, !stream.resource<*>{%c4}) -> !stream.resource<*>{%c4}
    // CHECK: scf.yield %{{.+}}, %[[DISPATCH]] : i32, !stream.resource<external>
    scf.yield %add, %disp : i32, !stream.resource<*>
  }
  // CHECK: util.return %[[WHILE]]#0, %[[WHILE]]#1 : i32, !stream.resource<external>
  util.return %while#0, %while#1 : i32, !stream.resource<*>
}

// -----

// Tests scf.while with type-changing dispatch in condition region. The dispatch
// produces transient which must transfer to staging for stream.async.load.
// Same-type transfer at end (external->external) is preserved.

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
  // CHECK: %[[RESULT_TRANSFER:.+]] = stream.async.transfer %[[WHILE]]#0 : !stream.resource<external>{%[[WHILE]]#1} -> !stream.resource<external>{%[[WHILE]]#1}
  %transfer = stream.async.transfer %while#0 : !stream.resource<*>{%while#1} -> !stream.resource<external>{%while#1}

  // CHECK: util.return %[[RESULT_TRANSFER]] : !stream.resource<external>
  util.return %transfer : !stream.resource<external>
}

// -----

// Tests scf.for with iter_args. The loop body uses transient resources for
// intermediate computation, with external input and output. Same-type transfer
// at end (external->external) is preserved.

// CHECK-LABEL: @testForOp
// CHECK-SAME: %[[ARG0:.+]]: index
// CHECK-SAME: %[[ARG1:.+]]: !stream.resource<external>
// CHECK-SAME: -> !stream.resource<external>
util.func public @testForOp(%arg0 : index, %arg1 : !stream.resource<*>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[C1:.+]] = arith.constant 1 : index
  // CHECK: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: %[[DISP0:.+]] = stream.async.dispatch @dispatch0(%[[ARG1]][%[[C0]] to %[[ARG0]] for %[[ARG0]]]) : (!stream.resource<external>{%[[C4]]}) -> !stream.resource<transient>{%[[C4]]}
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
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[DISP4]] : !stream.resource<external>{%[[ARG0]]} -> !stream.resource<external>{%[[ARG0]]}
  %transfer = stream.async.transfer %dispatch5 : !stream.resource<*>{%arg0} -> !stream.resource<external>{%arg0}

  // CHECK: util.return %[[TRANSFER]] : !stream.resource<external>
  util.return %transfer : !stream.resource<external>
}

// -----

// Tests that constant resources stored to globals preserve their constant
// lifetime even when they have external usage (e.g., being returned).

util.global private mutable @constant_global : !stream.resource<constant>

// CHECK-LABEL: @constant_global_with_external_use
util.func public @constant_global_with_external_use() -> !stream.resource<*> {
  %c4 = arith.constant 4 : index
  // CHECK: %[[CONSTANT:.+]] = stream.async.constant : !stream.resource<constant>{%[[C4:.+]]}
  %constant = stream.async.constant : !stream.resource<*>{%c4} = dense<1.0> : tensor<f32>
  // Transfer to ensure the value can be used as both constant (for store) and external (for return).
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[CONSTANT]] : !stream.resource<constant>{%[[C4]]} -> !stream.resource<constant>{%[[C4]]}
  %0 = stream.async.transfer %constant : !stream.resource<*>{%c4} -> !stream.resource<constant>{%c4}
  // CHECK: util.global.store %[[TRANSFER]], @constant_global : !stream.resource<constant>
  util.global.store %0, @constant_global : !stream.resource<constant>
  // CHECK: util.return %[[CONSTANT]] : !stream.resource<constant>
  util.return %constant : !stream.resource<*>
}

// -----

// Tests that non-constant globals correctly become External when used
// externally, and the Global+Constant special case doesn't interfere.

util.global private mutable @variable_global : !stream.resource<variable>

// CHECK-LABEL: @variable_global_with_external_use
util.func public @variable_global_with_external_use() -> !stream.resource<*> {
  %c4 = arith.constant 4 : index
  // CHECK: %[[VAR:.+]] = stream.async.alloca : !stream.resource<external>{%[[C4:.+]]}
  %var = stream.async.alloca : !stream.resource<*>{%c4}
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[VAR]] : !stream.resource<external>{%[[C4]]} -> !stream.resource<variable>{%[[C4]]}
  %0 = stream.async.transfer %var : !stream.resource<*>{%c4} -> !stream.resource<variable>{%c4}
  // CHECK: util.global.store %[[TRANSFER]], @variable_global : !stream.resource<variable>
  util.global.store %0, @variable_global : !stream.resource<variable>
  // CHECK: util.return %[[VAR]] : !stream.resource<external>
  util.return %var : !stream.resource<*>
}

// -----

// Tests that the Global+Constant special case ONLY applies when Global bit
// is set - constants used externally but NOT stored to globals should become
// External.

// CHECK-LABEL: @constant_external_without_global
util.func public @constant_external_without_global() -> !stream.resource<*> {
  %c4 = arith.constant 4 : index
  // CHECK: %[[CONST:.+]] = stream.async.constant : !stream.resource<external>{%[[C4:.+]]}
  %const = stream.async.constant : !stream.resource<*>{%c4} = dense<1.0> : tensor<f32>
  // CHECK: util.return %[[CONST]] : !stream.resource<external>
  util.return %const : !stream.resource<*>
}

// -----

// Tests that staging operations don't interfere with the Global+Constant
// priority.

util.global private mutable @constant_global_staging : !stream.resource<constant>

// CHECK-LABEL: @constant_global_with_staging
util.func public @constant_global_with_staging(%arg0: !stream.resource<staging>) -> !stream.resource<staging> {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  // CHECK: %[[CONST:.+]] = stream.async.constant : !stream.resource<constant>{%[[C4:.+]]}
  %const = stream.async.constant : !stream.resource<*>{%c4} = dense<2.0> : tensor<f32>
  // CHECK: %[[STAGING:.+]] = stream.async.transfer %[[CONST]] : !stream.resource<constant>{%[[C4]]} -> !stream.resource<staging>{%[[C4]]}
  %0 = stream.async.transfer %const : !stream.resource<*>{%c4} -> !stream.resource<staging>{%c4}
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[CONST]] : !stream.resource<constant>{%[[C4]]} -> !stream.resource<constant>{%[[C4]]}
  %1 = stream.async.transfer %const : !stream.resource<*>{%c4} -> !stream.resource<constant>{%c4}
  // CHECK: util.global.store %[[TRANSFER]], @constant_global_staging : !stream.resource<constant>
  util.global.store %1, @constant_global_staging : !stream.resource<constant>
  // CHECK: util.return %[[STAGING]]
  util.return %0 : !stream.resource<staging>
}

// -----

// Tests that constant globals preserve Constant lifetime even when used
// externally (copy-on-write optimization).

util.global private @const_weights : !stream.resource<constant>

// CHECK-LABEL: @load_constant_global_external_use
util.func public @load_constant_global_external_use() -> !stream.resource<constant> {
  // CHECK: %[[LOADED:.+]] = util.global.load @const_weights : !stream.resource<constant>
  %loaded = util.global.load @const_weights : !stream.resource<constant>
  // Should stay constant despite external use.
  // CHECK: util.return %[[LOADED]] : !stream.resource<constant>
  util.return %loaded : !stream.resource<constant>
}

// -----

// Tests that constant globals preserve Constant lifetime when transferred
// to staging (should insert transfer, not change source lifetime).

util.global private @const_data : !stream.resource<constant>

// CHECK-LABEL: @load_constant_global_staging_use
util.func public @load_constant_global_staging_use() -> !stream.resource<staging> {
  %c4 = arith.constant 4 : index
  // CHECK: %[[LOADED:.+]] = util.global.load @const_data : !stream.resource<constant>
  %loaded = util.global.load @const_data : !stream.resource<constant>
  // CHECK: %[[STAGING:.+]] = stream.async.transfer %[[LOADED]] : !stream.resource<constant>{%c4} -> !stream.resource<staging>{%c4}
  %staging = stream.async.transfer %loaded : !stream.resource<constant>{%c4} -> !stream.resource<staging>{%c4}
  // CHECK: util.return %[[STAGING]] : !stream.resource<staging>
  util.return %staging : !stream.resource<staging>
}

// -----

// Tests that variable globals correctly become External when used externally
// (no GlobalStorage+Constant optimization should apply).

util.global private mutable @var_state : !stream.resource<variable>

// CHECK-LABEL: @load_variable_global_external_use
util.func public @load_variable_global_external_use() -> !stream.resource<external> {
  %c4 = arith.constant 4 : index
  // CHECK: %[[LOADED:.+]] = util.global.load @var_state : !stream.resource<variable>
  %loaded = util.global.load @var_state : !stream.resource<variable>
  // Should become External (mutable data used externally).
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[LOADED]] : !stream.resource<variable>{%c4} -> !stream.resource<external>{%c4}
  %result = stream.async.transfer %loaded : !stream.resource<variable>{%c4} -> !stream.resource<external>{%c4}
  // CHECK: util.return %[[TRANSFER]] : !stream.resource<external>
  util.return %result : !stream.resource<external>
}

// -----

// Tests that globals loaded into control flow preserve storage identity.

util.global private @const_model : !stream.resource<constant>

// CHECK-LABEL: @load_constant_global_control_flow
util.func public @load_constant_global_control_flow(%cond: i1) -> !stream.resource<constant> {
  // CHECK: %[[LOADED:.+]] = util.global.load @const_model : !stream.resource<constant>
  %loaded = util.global.load @const_model : !stream.resource<constant>
  %c4 = arith.constant 4 : index
  // CHECK: %[[RESULT:.+]] = scf.if %{{.+}} -> (!stream.resource<constant>)
  %result = scf.if %cond -> !stream.resource<constant> {
    // CHECK: scf.yield %[[LOADED]] : !stream.resource<constant>
    scf.yield %loaded : !stream.resource<constant>
  } else {
    // CHECK: %[[OTHER:.+]] = stream.async.constant : !stream.resource<constant>
    %other = stream.async.constant : !stream.resource<constant>{%c4} = dense<0.0> : tensor<f32>
    // CHECK: scf.yield %[[OTHER]] : !stream.resource<constant>
    scf.yield %other : !stream.resource<constant>
  }
  // CHECK: util.return %[[RESULT]] : !stream.resource<constant>
  util.return %result : !stream.resource<constant>
}

// -----

// Tests that stream.resource.transients correctly refines both input and output types.
// When the result is external, the input resource must also be refined to external
// to satisfy the AllTypesMatch constraint.

// CHECK-LABEL: @transients_external_result
util.func public @transients_external_result(%size: index, %storage_size: index) -> !stream.resource<*> {
  %c123_i32 = arith.constant 123 : i32
  // CHECK: stream.async.splat {{.+}} -> !stream.resource<external>
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}

  // CHECK: stream.resource.alloc uninitialized : !stream.resource<transient>
  %storage = stream.resource.alloc uninitialized : !stream.resource<transient>{%storage_size}

  %immediate = stream.timepoint.immediate => !stream.timepoint

  // CHECK: stream.resource.transients {{.+}} !stream.resource<external>{{.+}} from {{.+}} !stream.resource<transient>
  %result, %result_tp = stream.resource.transients await(%immediate) => %splat : !stream.resource<*>{%size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // CHECK: stream.timepoint.await {{.+}} !stream.resource<external>
  %awaited = stream.timepoint.await %result_tp => %result : !stream.resource<*>{%size}

  // CHECK: util.return {{.+}} !stream.resource<external>
  util.return %awaited : !stream.resource<*>
}

// -----

// Tests that stream.async.cast propagates its constraint to the source value.
// The cast asserts the result must be external, so the source (splat) must also
// become external. After refinement, the cast folds away since types match.

// CHECK-LABEL: @asyncCastPropagation
util.func public @asyncCastPropagation(%size: index) -> !stream.resource<external> {
  %c123_i32 = arith.constant 123 : i32
  // The splat should be refined to external due to the cast constraint.
  // CHECK: %[[SPLAT:.+]] = stream.async.splat %c123_i32 {{.+}} -> !stream.resource<external>
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // The cast should fold away after the source type is refined to match.
  // CHECK-NOT: stream.async.cast
  %cast = stream.async.cast %splat : !stream.resource<*>{%size} -> !stream.resource<external>{%size}
  // CHECK: util.return %[[SPLAT]] : !stream.resource<external>
  util.return %cast : !stream.resource<external>
}
