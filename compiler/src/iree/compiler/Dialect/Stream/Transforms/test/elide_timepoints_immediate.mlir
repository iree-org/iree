// RUN: iree-opt --split-input-file --iree-stream-elide-timepoints %s | FileCheck %s

// Tests that joins with multiple immediate timepoints are marked as immediate.

// CHECK-LABEL: @immediateJoin
util.func private @immediateJoin() -> !stream.timepoint {
  %imm0 = stream.timepoint.immediate => !stream.timepoint
  %imm1 = stream.timepoint.immediate => !stream.timepoint
  // CHECK: stream.timepoint.join
  // CHECK-NEXT: %[[JOIN_IMM:.+]] = stream.timepoint.immediate
  %0 = stream.timepoint.join max(%imm0, %imm1) => !stream.timepoint
  // CHECK: util.return %[[JOIN_IMM]]
  util.return %0 : !stream.timepoint
}

// -----

// Tests that joins with one or more non-immediate timepoints are not elided.

// CHECK-LABEL: @nonImmediateJoin
// CHECK-SAME: (%[[NON_IMM:.+]]: !stream.timepoint)
util.func public @nonImmediateJoin(%arg0: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[NON_IMM]], %[[IMM]])
  %0 = stream.timepoint.join max(%arg0, %imm) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %0 : !stream.timepoint
}

// -----

// Tests that a select between immediate values is marked immediate.

// CHECK-LABEL: @selectSame
util.func public @selectSame(%cond: i1) -> !stream.timepoint {
  %imm0 = stream.timepoint.immediate => !stream.timepoint
  %imm1 = stream.timepoint.immediate => !stream.timepoint
  // CHECK: arith.select
  // CHECK-NEXT: %[[SELECT_IMM:.+]] = stream.timepoint.immediate
  %select = arith.select %cond, %imm0, %imm1 : !stream.timepoint
  // CHECK: util.return %[[SELECT_IMM]]
  util.return %select : !stream.timepoint
}

// -----

// Tests that a select with one or more unknown value is not marked immediate.

// CHECK-LABEL: @selectDifferent
util.func public @selectDifferent(%cond: i1, %unknown: !stream.timepoint) -> !stream.timepoint {
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK: %[[SELECT:.+]] = arith.select
  %select = arith.select %cond, %imm, %unknown : !stream.timepoint
  // CHECK: util.return %[[SELECT]]
  util.return %select : !stream.timepoint
}

// -----

// Tests global immediate timepoints are marked immediate when loaded.

util.global private mutable @global = #stream.timepoint<immediate> : !stream.timepoint

// CHECK-LABEL: @immediateGlobal
util.func private @immediateGlobal() -> !stream.timepoint {
  // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
  %global = util.global.load @global : !stream.timepoint
  // CHECK: util.return %[[IMM]]
  util.return %global : !stream.timepoint
}

// -----

// Tests that uniform global store->load forwarding handles immediates.

util.global private mutable @global : !stream.timepoint

// CHECK-LABEL: @uniformGlobal
util.func private @uniformGlobal() -> !stream.timepoint {
  %imm = stream.timepoint.immediate => !stream.timepoint
  util.global.store %imm, @global : !stream.timepoint
  // CHECK: util.global.load
  %global = util.global.load @global : !stream.timepoint
  // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
  // CHECK: util.return %[[IMM]]
  util.return %global : !stream.timepoint
}
util.func private @globalSetter() {
  %imm = stream.timepoint.immediate => !stream.timepoint
  util.global.store %imm, @global : !stream.timepoint
  util.return
}

// -----

// Tests that divergent global stores do not propagate.

util.global private mutable @global = #stream.timepoint<immediate> : !stream.timepoint

// CHECK-LABEL: @nonUniformGlobal
util.func private @nonUniformGlobal() -> !stream.timepoint {
  // CHECK-NOT: stream.timepoint.immediate
  // CHECK: %[[GLOBAL:.+]] = util.global.load @global
  %global = util.global.load @global : !stream.timepoint
  // CHECK: util.return %[[GLOBAL]]
  util.return %global : !stream.timepoint
}
util.func public @globalSetter(%arg0: !stream.timepoint) {
  util.global.store %arg0, @global : !stream.timepoint
  util.return
}

// -----

// Tests that meaningful timeline ops are never marked immediate.

// CHECK-LABEL: @nonImmediate
util.func private @nonImmediate() -> !stream.timepoint {
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK: %[[EXECUTE:.+]] = stream.cmd.execute
  %0 = stream.cmd.execute await(%imm) => with() {} => !stream.timepoint
  // CHECK: util.return %[[EXECUTE]]
  util.return %0 : !stream.timepoint
}

// -----

// Tests that an immediate timepoint passed along a call edge is propagated.

// CHECK-LABEL: util.func public @caller
util.func public @caller() -> !stream.timepoint {
  // CHECK: %[[T0_IMM:.+]] = stream.timepoint.immediate
  %t0 = stream.timepoint.immediate => !stream.timepoint
  // CHECK: %[[T1:.+]] = util.call @callee(%[[T0_IMM]], %[[T0_IMM]])
  // CHECK-NEXT: %[[T1_IMM:.+]] = stream.timepoint.immediate
  %t1 = util.call @callee(%t0, %t0) : (!stream.timepoint, !stream.timepoint) -> !stream.timepoint
  // CHECK: %[[T2:.+]] = stream.timepoint.join max(%[[T0_IMM]], %[[T1_IMM]])
  // CHECK-NEXT: %[[T2_IMM:.+]] = stream.timepoint.immediate
  %t2 = stream.timepoint.join max(%t0, %t1) => !stream.timepoint
  // CHECK: util.return %[[T2_IMM]]
  util.return %t2 : !stream.timepoint
}
// CHECK-LABEL: util.func private @callee
util.func private @callee(%t0a: !stream.timepoint, %t0b: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[T0A_IMM:.+]] = stream.timepoint.immediate
  // CHECK: %[[T0B_IMM:.+]] = stream.timepoint.immediate
  // CHECK-NEXT: %[[T1:.+]] = stream.timepoint.join max(%[[T0A_IMM]], %[[T0B_IMM]])
  %t1 = stream.timepoint.join max(%t0a, %t0b) => !stream.timepoint
  // CHECK-NEXT: %[[T1_IMM:.+]] = stream.timepoint.immediate
  // CHECK-NEXT: util.return %[[T1_IMM]]
  util.return %t1 : !stream.timepoint
}

// -----

// Tests that duplicate call args/results are handled correctly.

// CHECK-LABEL: util.func public @callerDupes
util.func public @callerDupes() -> !stream.timepoint {
  // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK: %[[CALL:.+]]:2 = util.call @calleeDupes
  // CHECK-NEXT: %[[CALL_IMM0:.+]] = stream.timepoint.immediate
  // CHECK-NEXT: %[[CALL_IMM1:.+]] = stream.timepoint.immediate
  %call:2 = util.call @calleeDupes(%imm, %imm) : (!stream.timepoint, !stream.timepoint) -> (!stream.timepoint, !stream.timepoint)
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[CALL_IMM0]], %[[CALL_IMM1]])
  // CHECK-NEXT: %[[JOIN_IMM:.+]] = stream.timepoint.immediate
  %join = stream.timepoint.join max(%call#0, %call#1) => !stream.timepoint
  // CHECK: util.return %[[JOIN_IMM]]
  util.return %join : !stream.timepoint
}
util.func private @calleeDupes(%arg0: !stream.timepoint, %arg1: !stream.timepoint) -> (!stream.timepoint, !stream.timepoint) {
  util.return %arg0, %arg1 : !stream.timepoint, !stream.timepoint
}

// -----

// Tests that convergent caller timepoints are handled correctly.

// CHECK-LABEL: util.func public @uniformCaller
util.func public @uniformCaller() -> !stream.timepoint {
  // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK-NEXT: call @uniformCallee(%[[IMM]])
  // CHECK-NEXT: %[[CALL_IMM0:.+]] = stream.timepoint.immediate
  %call0 = util.call @uniformCallee(%imm) : (!stream.timepoint) -> !stream.timepoint
  // CHECK-NEXT: call @uniformCallee(%[[IMM]])
  // CHECK-NEXT: %[[CALL_IMM1:.+]] = stream.timepoint.immediate
  %call1 = util.call @uniformCallee(%imm) : (!stream.timepoint) -> !stream.timepoint
  // CHECK-NEXT: %[[CALLER_JOIN:.+]] = stream.timepoint.join max(%[[CALL_IMM0]], %[[CALL_IMM1]])
  // CHECK-NEXT: %[[CALLER_JOIN_IMM:.+]] = stream.timepoint.immediate
  %join = stream.timepoint.join max(%call0, %call1) => !stream.timepoint
  // CHECK: util.return %[[CALLER_JOIN_IMM]]
  util.return %join : !stream.timepoint
}
// CHECK: util.func private @uniformCallee
util.func private @uniformCallee(%arg0: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[ARG0_IMM:.+]] = stream.timepoint.immediate
  // CHECK: %[[CALLEE_JOIN:.+]] = stream.timepoint.join max(%[[ARG0_IMM]])
  // CHECK-NEXT: %[[CALLEE_JOIN_IMM:.+]] = stream.timepoint.immediate
  %0 = stream.timepoint.join max(%arg0) => !stream.timepoint
  // CHECK: util.return %[[CALLEE_JOIN_IMM]]
  util.return %0 : !stream.timepoint
}

// -----

// Tests that divergent caller timepoints are handled correctly.
// NOTE: if we ever implemented execution tracing we could discover that %call1
// should be immediate - today, though, we aggregate over callers and any one
// that may pass a non-immediate poisons the analysis.

// CHECK-LABEL: util.func public @nonUniformCaller
// CHECK-SAME: (%[[UNKNOWN:.+]]: !stream.timepoint)
util.func public @nonUniformCaller(%unknown: !stream.timepoint) -> !stream.timepoint {
  // CHECK-NOT: stream.timepoint.immediate
  // CHECK: %[[CALL0:.+]] = util.call @nonUniformCallee(%[[UNKNOWN]])
  %call0 = util.call @nonUniformCallee(%unknown) : (!stream.timepoint) -> !stream.timepoint
  // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK: %[[CALL1:.+]] = util.call @nonUniformCallee(%[[IMM]])
  %call1 = util.call @nonUniformCallee(%imm) : (!stream.timepoint) -> !stream.timepoint
  // CHECK: %[[CALLER_JOIN:.+]] = stream.timepoint.join max(%[[CALL0]], %[[CALL1]])
  %join = stream.timepoint.join max(%call0, %call1) => !stream.timepoint
  // CHECK: util.return %[[CALLER_JOIN]]
  util.return %join : !stream.timepoint
}
// CHECK-LABEL: util.func private @nonUniformCallee
// CHECK-SAME: (%[[CALLEE_ARG:.+]]: !stream.timepoint)
util.func private @nonUniformCallee(%arg0: !stream.timepoint) -> !stream.timepoint {
  // CHECK-NOT: stream.timepoint.immediate
  // CHECK: %[[CALLEE_JOIN:.+]] = stream.timepoint.join max(%[[CALLEE_ARG]])
  %0 = stream.timepoint.join max(%arg0) => !stream.timepoint
  // CHECK: util.return %[[CALLEE_JOIN]]
  util.return %0 : !stream.timepoint
}

// -----

// Tests that an immediate timepoint passed along a block edge is propagated.

// CHECK-LABEL: util.func public @branch
util.func public @branch() -> !stream.timepoint {
  %t0 = stream.timepoint.immediate => !stream.timepoint
  // CHECK: cf.br ^bb1
  cf.br ^bb1(%t0 : !stream.timepoint)
// CHECK-NEXT: ^bb1(%[[BB1_T0:.+]]: !stream.timepoint)
^bb1(%bb1_t0: !stream.timepoint):
  // CHECK-NEXT: %[[BB1_T0_IMMEDIATE:.+]] = stream.timepoint.immediate
  // CHECK-NEXT: %[[T1:.+]] = stream.timepoint.join max(%[[BB1_T0_IMMEDIATE]])
  %t1 = stream.timepoint.join max(%bb1_t0) => !stream.timepoint
  // CHECK-NEXT: %[[JOIN_IMMEDIATE:.+]] = stream.timepoint.immediate
  // CHECK-NEXT: util.return %[[JOIN_IMMEDIATE]]
  util.return %t1 : !stream.timepoint
}

// -----

// Tests that forward edges with convergently immediate timepoints get
// propagated.

// CHECK-LABEL: util.func public @branchConvergentForwardEdge
util.func public @branchConvergentForwardEdge(%cond: i1) -> !stream.timepoint {
  // CHECK: %[[IMM0:.+]] = stream.timepoint.immediate
  %imm0 = stream.timepoint.immediate => !stream.timepoint
  // CHECK: %[[IMM1:.+]] = stream.timepoint.immediate
  %imm1 = stream.timepoint.immediate => !stream.timepoint
  // CHECK-NEXT: cf.cond_br
  // CHECK-SAME:   ^bb1(%[[IMM0]] : !stream.timepoint),
  // CHECK-SAME:   ^bb1(%[[IMM1]] : !stream.timepoint)
  cf.cond_br %cond, ^bb1(%imm0 : !stream.timepoint), ^bb1(%imm1 : !stream.timepoint)
// CHECK-NEXT: ^bb1(%[[BB1_ARG:.+]]: !stream.timepoint)
^bb1(%bb1_arg: !stream.timepoint):
  // CHECK: %[[BB1_IMM:.+]] = stream.timepoint.immediate
  // CHECK: util.return %[[BB1_IMM]]
  util.return %bb1_arg : !stream.timepoint
}

// -----

// Tests that forward edges with divergent timepoints don't get propagated.

// CHECK-LABEL: util.func public @branchDivergentForwardEdge
// CHECK-SAME: (%[[COND:.+]]: i1, %[[UNKNOWN:.+]]: !stream.timepoint)
util.func public @branchDivergentForwardEdge(%cond: i1, %unknown: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK-NEXT: cf.cond_br %[[COND]]
  // CHECK-SAME:   ^bb1(%[[UNKNOWN]] : !stream.timepoint),
  // CHECK-SAME:   ^bb1(%[[IMM]] : !stream.timepoint)
  cf.cond_br %cond, ^bb1(%unknown : !stream.timepoint), ^bb1(%imm : !stream.timepoint)
// CHECK-NEXT: ^bb1(%[[BB1_ARG:.+]]: !stream.timepoint)
^bb1(%bb1_arg: !stream.timepoint):
  // CHECK: util.return %[[BB1_ARG]]
  util.return %bb1_arg : !stream.timepoint
}

// -----

// Tests that back edges with divergent timepoints don't get propagated.

// CHECK-LABEL: util.func public @branchDivergentBackEdge
util.func public @branchDivergentBackEdge(%cond: i1) -> !stream.timepoint {
  %t0 = stream.timepoint.immediate => !stream.timepoint
  // CHECK: cf.br ^bb1
  cf.br ^bb1(%cond, %t0 : i1, !stream.timepoint)
// CHECK-NEXT: ^bb1(%[[BB1_COND:.+]]: i1, %[[BB1_T0:.+]]: !stream.timepoint)
^bb1(%bb1_cond: i1, %bb1_t0: !stream.timepoint):
  // CHECK-NOT: stream.timepoint.immediate
  // CHECK-NEXT: %[[BB1_T1:.+]] = stream.cmd.execute await(%[[BB1_T0]])
  %bb1_t1 = stream.cmd.execute await(%bb1_t0) => with() {} => !stream.timepoint
  // CHECK: %[[FALSE:.+]] = arith.constant false
  %cond_false = arith.constant false
  // CHECK-NEXT: cf.cond_br
  // CHECK-SAME:   ^bb1(%[[FALSE]], %[[BB1_T1]] : i1, !stream.timepoint)
  // CHECK-SAME:   ^bb2(%[[BB1_T1]] : !stream.timepoint)
  cf.cond_br %bb1_cond, ^bb1(%cond_false, %bb1_t1 : i1, !stream.timepoint), ^bb2(%bb1_t1 : !stream.timepoint)
// CHECK-NEXT: ^bb2(%[[BB2_T1:.+]]: !stream.timepoint)
^bb2(%bb2_t1: !stream.timepoint):
  // CHECK-NEXT: util.return %[[BB2_T1]]
  util.return %bb2_t1 : !stream.timepoint
}

// -----

// Tests that scf.if regions with convergent yields are handled.

// CHECK-LABEL: util.func public @scfIfConvergent
// CHECK-SAME: (%[[COND:.+]]: i1)
util.func public @scfIfConvergent(%cond: i1) -> !stream.timepoint {
  // CHECK: %[[IF:.+]] = scf.if
  %if = scf.if %cond -> !stream.timepoint {
    // CHECK: %[[IMM0:.+]] = stream.timepoint.immediate
    %imm0 = stream.timepoint.immediate => !stream.timepoint
    // CHECK: yield %[[IMM0]]
    scf.yield %imm0 : !stream.timepoint
  } else {
    // CHECK: %[[IMM1:.+]] = stream.timepoint.immediate
    %imm1 = stream.timepoint.immediate => !stream.timepoint
    // CHECK: yield %[[IMM1]]
    scf.yield %imm1 : !stream.timepoint
  }
  // CHECK: %[[IF_IMM:.+]] = stream.timepoint.immediate
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[IF_IMM]])
  // CHECK-NEXT: %[[JOIN_IMM:.+]] = stream.timepoint.immediate
  %join = stream.timepoint.join max(%if) => !stream.timepoint
  // CHECK: util.return %[[JOIN_IMM]]
  util.return %join : !stream.timepoint
}

// -----

// Tests that scf.if regions with divergent yields are handled.

// CHECK-LABEL: util.func public @scfIfDivergent
// CHECK-SAME: (%[[COND:.+]]: i1, %[[UNKNOWN:.+]]: !stream.timepoint)
util.func public @scfIfDivergent(%cond: i1, %unknown: !stream.timepoint) -> !stream.timepoint {
  // CHECK: %[[IMM:.+]] = stream.timepoint.immediate
  %imm = stream.timepoint.immediate => !stream.timepoint
  // CHECK: %[[IF:.+]] = scf.if
  %0 = scf.if %cond -> !stream.timepoint {
    // CHECK: yield %[[IMM]]
    scf.yield %imm : !stream.timepoint
  } else {
    // CHECK: %[[JOIN1:.+]] = stream.timepoint.join max(%[[UNKNOWN]], %[[IMM]])
    %join1 = stream.timepoint.join max(%unknown, %imm) => !stream.timepoint
    // CHECK: yield %[[JOIN1]]
    scf.yield %join1 : !stream.timepoint
  }
  // CHECK-NOT: stream.timepoint.immediate
  // CHECK: %[[JOIN_OUTER:.+]] = stream.timepoint.join max(%[[UNKNOWN]], %[[IF]])
  %join_outer = stream.timepoint.join max(%unknown, %0) => !stream.timepoint
  // CHECK: util.return %[[JOIN_OUTER]]
  util.return %join_outer : !stream.timepoint
}

// TODO(benvanik): support scf.for
