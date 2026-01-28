// RUN: iree-opt --split-input-file --iree-util-propagate-subranges %s | FileCheck %s

// Tests that resource global loads also load all the subrange params.
//
// This rotates subranges through stores and into loads.

//      CHECK: util.global private mutable @constantGlobal : !util.buffer
// CHECK-NEXT: util.global private mutable @constantGlobal__storage_size : index
// CHECK-NEXT: util.global private mutable @constantGlobal__offset : index
// CHECK-NEXT: util.global private mutable @constantGlobal__length : index
util.global private mutable @constantGlobal : !util.buffer

// CHECK-LABEL: @globalLoad
util.func private @globalLoad() {
  // CHECK-NEXT: %[[RESOURCE:.+]] = util.global.load @constantGlobal : !util.buffer
  // CHECK-NEXT: %[[STORAGE_SIZE:.+]] = util.global.load @constantGlobal__storage_size : index
  // CHECK-NEXT: %[[OFFSET:.+]] = util.global.load @constantGlobal__offset : index
  // CHECK-NEXT: %[[LENGTH:.+]] = util.global.load @constantGlobal__length : index
  // CHECK: %[[SUBRANGE:.+]] = util.buffer.subspan %[[RESOURCE]][%[[OFFSET]]] : !util.buffer{%[[STORAGE_SIZE]]} -> !util.buffer{%[[LENGTH]]}
  %0 = util.global.load @constantGlobal : !util.buffer
  // CHECK-NEXT: util.optimization_barrier %[[SUBRANGE]]
  util.optimization_barrier %0 : !util.buffer
  util.return
}

// -----

// Tests that resource global stores consume their incoming subranges.
//
// This rotates subranges through stores and into loads.

//      CHECK: util.global private mutable @mutableGlobal : !util.buffer
// CHECK-NEXT: util.global private mutable @mutableGlobal__storage_size : index
// CHECK-NEXT: util.global private mutable @mutableGlobal__offset : index
// CHECK-NEXT: util.global private mutable @mutableGlobal__length : index
util.global private mutable @mutableGlobal : !util.buffer

// CHECK-LABEL: @globalStore
// CHECK-SAME: (%[[RESOURCE:.+]]: !util.buffer, %[[STORAGE_SIZE:.+]]: index, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index)
util.func private @globalStore(%resource: !util.buffer) {
  // CHECK: util.global.store %[[RESOURCE]], @mutableGlobal : !util.buffer
  // CHECK: util.global.store %[[STORAGE_SIZE]], @mutableGlobal__storage_size : index
  // CHECK: util.global.store %[[OFFSET]], @mutableGlobal__offset : index
  // CHECK: util.global.store %[[LENGTH]], @mutableGlobal__length : index
  util.global.store %resource, @mutableGlobal : !util.buffer
  util.return
}

// -----

// Tests that function arguments are expanded into an explicit subrange of
// (resource, size, offset, length).
//
// This rotates subranges from callers into callees.

// CHECK-LABEL: @funcArgs
// CHECK-SAME: (%[[RESOURCE0:.+]]: !util.buffer, %[[STORAGE_SIZE0:.+]]: index, %[[OFFSET0:.+]]: index, %[[LENGTH0:.+]]: index, %[[RESOURCE1:.+]]: !util.buffer, %[[STORAGE_SIZE1:.+]]: index, %[[OFFSET1:.+]]: index, %[[LENGTH1:.+]]: index)
util.func private @funcArgs(%resource0: !util.buffer, %resource1: !util.buffer) {
  // CHECK-NEXT: %[[SUBRANGE0:.+]] = util.buffer.subspan %[[RESOURCE0]][%[[OFFSET0]]] : !util.buffer{%[[STORAGE_SIZE0]]} -> !util.buffer{%[[LENGTH0]]}
  // CHECK-NEXT: %[[SUBRANGE1:.+]] = util.buffer.subspan %[[RESOURCE1]][%[[OFFSET1]]] : !util.buffer{%[[STORAGE_SIZE1]]} -> !util.buffer{%[[LENGTH1]]}

  // CHECK-NEXT: util.optimization_barrier %[[SUBRANGE0]]
  util.optimization_barrier %resource0 : !util.buffer
  // CHECK-NEXT: util.optimization_barrier %[[SUBRANGE1]]
  util.optimization_barrier %resource1 : !util.buffer

  util.return
}

// -----

// Tests that function results are expanded into an explicit subrange of
// (resource, size, offset, length).
//
// This rotates subranges from callees into callers.

// CHECK-LABEL: @funcResults
// CHECK-SAME: (%[[RESOURCE0:.+]]: !util.buffer, %[[STORAGE_SIZE0:.+]]: index, %[[OFFSET0:.+]]: index, %[[LENGTH0:.+]]: index, %[[RESOURCE1:.+]]: !util.buffer, %[[STORAGE_SIZE1:.+]]: index, %[[OFFSET1:.+]]: index, %[[LENGTH1:.+]]: index)
// CHECK-SAME: -> (!util.buffer, index, index, index, !util.buffer, index, index, index)
util.func private @funcResults(%resource0: !util.buffer, %resource1: !util.buffer) -> (!util.buffer, !util.buffer) {
  // NOTE: there will be extra stuff here from the arg insertion. Since the
  // return should consume the subrange that was inserted we expect to directly
  // use the function arguments.

  // CHECK: util.return %[[RESOURCE0]], %[[STORAGE_SIZE0]], %[[OFFSET0]], %[[LENGTH0]], %[[RESOURCE1]], %[[STORAGE_SIZE1]], %[[OFFSET1]], %[[LENGTH1]]
  util.return %resource0, %resource1 : !util.buffer, !util.buffer
}


// -----

// Tests that exported functions don't have their signature changed.

// CHECK-LABEL: @publicFuncSignature
// CHECK-SAME: (%[[RESOURCE:.+]]: !util.buffer) -> !util.buffer
util.func @publicFuncSignature(%resource: !util.buffer) -> !util.buffer {
  // CHECK-NEXT: util.return %[[RESOURCE]] : !util.buffer
  util.return %resource : !util.buffer
}

// -----

// Tests that function calls have their args and results expanded into
// (resource, size, offset, length).
//
// This rotates subranges on args from callers to callees and subranges on results
// from callees to callers.

// CHECK-LABEL: @caller
// CHECK-SAME: (%[[RESOURCE0:.+]]: !util.buffer, %[[STORAGE_SIZE0:.+]]: index, %[[OFFSET0:.+]]: index, %[[LENGTH0:.+]]: index, %[[RESOURCE1:.+]]: !util.buffer, %[[STORAGE_SIZE1:.+]]: index, %[[OFFSET1:.+]]: index, %[[LENGTH1:.+]]: index)
util.func private @caller(%resource0: !util.buffer, %resource1: !util.buffer) {
  // NOTE: there will be extra stuff here from the arg insertion. The call
  // consumes the subranges and we expect the args to be passed directly.

  // CHECK: %[[RET:.+]]:8 = util.call @callee(%[[RESOURCE0]], %[[STORAGE_SIZE0]], %[[OFFSET0]], %[[LENGTH0]], %[[RESOURCE1]], %[[STORAGE_SIZE1]], %[[OFFSET1]], %[[LENGTH1]]) : (!util.buffer, index, index, index, !util.buffer, index, index, index) -> (!util.buffer, index, index, index, !util.buffer, index, index, index)
  %0:2 = util.call @callee(%resource0, %resource1) : (!util.buffer, !util.buffer) -> (!util.buffer, !util.buffer)
  // CHECK-NEXT: %[[RET_SUBRANGE0:.+]] = util.buffer.subspan %[[RET]]#0[%[[RET]]#2] : !util.buffer{%[[RET]]#1} -> !util.buffer{%[[RET]]#3}
  // CHECK-NEXT: %[[RET_SUBRANGE1:.+]] = util.buffer.subspan %[[RET]]#4[%[[RET]]#6] : !util.buffer{%[[RET]]#5} -> !util.buffer{%[[RET]]#7}

  // CHECK-NEXT: util.optimization_barrier %[[RET_SUBRANGE0]] : !util.buffer
  util.optimization_barrier %0#0 : !util.buffer
  // CHECK-NEXT: util.optimization_barrier %[[RET_SUBRANGE1]] : !util.buffer
  util.optimization_barrier %0#1 : !util.buffer

  util.return
}

util.func private @callee(%arg0: !util.buffer, %arg1: !util.buffer) -> (!util.buffer, !util.buffer) {
  util.return %arg0, %arg1 : !util.buffer, !util.buffer
}

// -----

// Tests that scf.if operations propagate subranges through both branches.

// CHECK-LABEL: @scfIfWithResults
// CHECK-SAME: (%[[COND:.+]]: i1, %[[ARG:.+]]: !util.buffer, %[[ARG_SIZE:.+]]: index, %[[ARG_OFFSET:.+]]: index, %[[ARG_LENGTH:.+]]: index)
// CHECK-SAME: -> (!util.buffer, index, index, index)
util.func private @scfIfWithResults(%cond: i1, %arg: !util.buffer) -> !util.buffer {
  // Function args expanded, create subrange from expanded args.
  // CHECK: %[[ARG_SUBRANGE:.+]] = util.buffer.subspan %[[ARG]][%[[ARG_OFFSET]]] : !util.buffer{%[[ARG_SIZE]]} -> !util.buffer{%[[ARG_LENGTH]]}

  // scf.if result types expanded to 4 values.
  // CHECK: %[[IF_RESULT:.+]]:4 = scf.if %[[COND]] -> (!util.buffer, index, index, index) {
  %0 = scf.if %cond -> !util.buffer {
    // THEN branch: compute size/offset/length, create nested subspan.
    // CHECK-NEXT: %[[THEN_SIZE:.+]] = util.buffer.size %[[ARG_SUBRANGE]]
    %then_size = util.buffer.size %arg : !util.buffer
    // CHECK-NEXT: %[[THEN_OFFSET:.+]] = arith.constant 10
    %then_offset = arith.constant 10 : index
    // CHECK-NEXT: %[[THEN_LENGTH:.+]] = arith.constant 20
    %then_length = arith.constant 20 : index
    // CHECK-NEXT: %[[THEN_SUBSPAN:.+]] = util.buffer.subspan %[[ARG_SUBRANGE]][%[[THEN_OFFSET]]] : !util.buffer{%[[THEN_SIZE]]} -> !util.buffer{%[[THEN_LENGTH]]}
    %then_subspan = util.buffer.subspan %arg[%then_offset] : !util.buffer{%then_size} -> !util.buffer{%then_length}
    // Adjusted offset computed but not used in yield (updateSubrangeOp tracking).
    // CHECK-NEXT: %[[ADJUSTED:.+]] = arith.addi %[[ARG_OFFSET]], %[[THEN_OFFSET]]
    // Yields: parent subrange, size, LOCAL offset, LOCAL length (not adjusted offset!).
    // CHECK-NEXT: scf.yield %[[ARG_SUBRANGE]], %[[THEN_SIZE]], %[[THEN_OFFSET]], %[[THEN_LENGTH]] : !util.buffer, index, index, index
    scf.yield %then_subspan : !util.buffer
  } else {
    // ELSE branch: yields original expanded args unchanged.
    // CHECK: scf.yield %[[ARG]], %[[ARG_SIZE]], %[[ARG_OFFSET]], %[[ARG_LENGTH]] : !util.buffer, index, index, index
    scf.yield %arg : !util.buffer
  }
  // After if: create subrange from if results (for compatibility).
  // CHECK: %[[RESULT_SUBRANGE:.+]] = util.buffer.subspan %[[IF_RESULT]]#0[%[[IF_RESULT]]#2] : !util.buffer{%[[IF_RESULT]]#1} -> !util.buffer{%[[IF_RESULT]]#3}
  // Return the 4-tuple directly (function signature already expanded).
  // CHECK: util.return %[[IF_RESULT]]#0, %[[IF_RESULT]]#1, %[[IF_RESULT]]#2, %[[IF_RESULT]]#3 : !util.buffer, index, index, index
  util.return %0 : !util.buffer
}

// -----

// Tests that scf.for operations propagate subranges through iter_args.

// CHECK-LABEL: @scfForWithIterArgs
// CHECK-SAME: (%[[ARG:.+]]: !util.buffer, %[[ARG_SIZE:.+]]: index, %[[ARG_OFFSET:.+]]: index, %[[ARG_LENGTH:.+]]: index)
util.func private @scfForWithIterArgs(%arg: !util.buffer) -> !util.buffer {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index

  // CHECK-DAG: %[[ARG_SUBRANGE:.+]] = util.buffer.subspan %[[ARG]][%[[ARG_OFFSET]]] : !util.buffer{%[[ARG_SIZE]]} -> !util.buffer{%[[ARG_LENGTH]]}
  // CHECK-DAG: %{{.+}} = arith.constant 0
  // CHECK-DAG: %{{.+}} = arith.constant 10
  // CHECK-DAG: %{{.+}} = arith.constant 1

  // CHECK: %[[FOR_RESULT:.+]]:4 = scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%{{.+}} = %[[ARG]], %{{.+}} = %[[ARG_SIZE]], %{{.+}} = %[[ARG_OFFSET]], %{{.+}} = %[[ARG_LENGTH]]) -> (!util.buffer, index, index, index)
  %0 = scf.for %i = %c0 to %c10 step %c1 iter_args(%iter = %arg) -> !util.buffer {
    // CHECK: util.buffer.size
    %next_size = util.buffer.size %iter : !util.buffer
    // CHECK: arith.constant 0
    %next_offset = arith.constant 0 : index
    // CHECK: scf.yield
    scf.yield %iter : !util.buffer
  }
  // CHECK: %[[RESULT_SUBRANGE:.+]] = util.buffer.subspan %[[FOR_RESULT]]#0[%[[FOR_RESULT]]#2] : !util.buffer{%[[FOR_RESULT]]#1} -> !util.buffer{%[[FOR_RESULT]]#3}
  // CHECK: util.return %[[FOR_RESULT]]#0, %[[FOR_RESULT]]#1, %[[FOR_RESULT]]#2, %[[FOR_RESULT]]#3
  util.return %0 : !util.buffer
}

// -----

// Tests that scf.while operations propagate subranges through before and after regions.

// CHECK-LABEL: @scfWhileWithResources
// CHECK-SAME: (%[[ARG:.+]]: !util.buffer, %[[ARG_SIZE:.+]]: index, %[[ARG_OFFSET:.+]]: index, %[[ARG_LENGTH:.+]]: index)
util.func private @scfWhileWithResources(%arg: !util.buffer) -> !util.buffer {
  %true = arith.constant true
  %false = arith.constant false

  // CHECK-DAG: %[[ARG_SUBRANGE:.+]] = util.buffer.subspan %[[ARG]][%[[ARG_OFFSET]]] : !util.buffer{%[[ARG_SIZE]]} -> !util.buffer{%[[ARG_LENGTH]]}
  // CHECK-DAG: %{{.+}} = arith.constant true
  // CHECK-DAG: %{{.+}} = arith.constant false

  // CHECK: %[[WHILE_RESULT:.+]]:4 = scf.while (%{{.+}} = %[[ARG]], %{{.+}} = %[[ARG_SIZE]], %{{.+}} = %[[ARG_OFFSET]], %{{.+}} = %[[ARG_LENGTH]]) : (!util.buffer, index, index, index) -> (!util.buffer, index, index, index)
  %0 = scf.while (%arg0 = %arg) : (!util.buffer) -> !util.buffer {
    // CHECK-NEXT: %{{.+}} = util.buffer.subspan
    // CHECK-NEXT: scf.condition
    scf.condition(%true) %arg0 : !util.buffer
  } do {
  ^bb0(%arg1: !util.buffer):
    // CHECK: ^bb0(%{{.+}}: !util.buffer, %{{.+}}: index, %{{.+}}: index, %{{.+}}: index):
    // CHECK-NEXT: %{{.+}} = util.buffer.subspan
    // CHECK-NEXT: %{{.+}} = util.buffer.size
    %next_size = util.buffer.size %arg1 : !util.buffer
    // CHECK-NEXT: %{{.+}} = arith.constant 0
    %next_offset = arith.constant 0 : index
    // CHECK-NEXT: scf.yield
    scf.yield %arg1 : !util.buffer
  }
  // CHECK: %[[RESULT_SUBRANGE:.+]] = util.buffer.subspan %[[WHILE_RESULT]]#0[%[[WHILE_RESULT]]#2] : !util.buffer{%[[WHILE_RESULT]]#1} -> !util.buffer{%[[WHILE_RESULT]]#3}
  // CHECK: util.return %[[WHILE_RESULT]]#0, %[[WHILE_RESULT]]#1, %[[WHILE_RESULT]]#2, %[[WHILE_RESULT]]#3
  util.return %0 : !util.buffer
}

// -----

// Tests that scf.index_switch operations propagate subranges through all cases.

// CHECK-LABEL: @scfIndexSwitchWithResources
// CHECK-SAME: (%[[IDX:.+]]: index, %[[ARG:.+]]: !util.buffer, %[[ARG_SIZE:.+]]: index, %[[ARG_OFFSET:.+]]: index, %[[ARG_LENGTH:.+]]: index)
util.func private @scfIndexSwitchWithResources(%idx: index, %arg: !util.buffer) -> !util.buffer {
  // CHECK: %[[ARG_SUBRANGE:.+]] = util.buffer.subspan %[[ARG]][%[[ARG_OFFSET]]] : !util.buffer{%[[ARG_SIZE]]} -> !util.buffer{%[[ARG_LENGTH]]}

  // CHECK: %[[SWITCH_RESULT:.+]]:4 = scf.index_switch %[[IDX]] -> !util.buffer, index, index, index
  %0 = scf.index_switch %idx -> !util.buffer
  case 0 {
    // CHECK: %[[CASE0_SIZE:.+]] = util.buffer.size %[[ARG_SUBRANGE]]
    %case0_size = util.buffer.size %arg : !util.buffer
    // CHECK: %[[CASE0_OFFSET:.+]] = arith.constant 5
    %case0_offset = arith.constant 5 : index
    // CHECK: %[[CASE0_LENGTH:.+]] = arith.constant 10
    %case0_length = arith.constant 10 : index
    // CHECK: %[[CASE0_SUBSPAN:.+]] = util.buffer.subspan %[[ARG_SUBRANGE]][%[[CASE0_OFFSET]]] : !util.buffer{%[[CASE0_SIZE]]} -> !util.buffer{%[[CASE0_LENGTH]]}
    %case0_subspan = util.buffer.subspan %arg[%case0_offset] : !util.buffer{%case0_size} -> !util.buffer{%case0_length}
    // CHECK: scf.yield %[[ARG_SUBRANGE]], %[[CASE0_SIZE]], %[[CASE0_OFFSET]], %[[CASE0_LENGTH]]
    scf.yield %case0_subspan : !util.buffer
  }
  case 1 {
    // CHECK: scf.yield %[[ARG]], %[[ARG_SIZE]], %[[ARG_OFFSET]], %[[ARG_LENGTH]]
    scf.yield %arg : !util.buffer
  }
  default {
    // CHECK: scf.yield %[[ARG]], %[[ARG_SIZE]], %[[ARG_OFFSET]], %[[ARG_LENGTH]]
    scf.yield %arg : !util.buffer
  }
  // CHECK: %[[RESULT_SUBRANGE:.+]] = util.buffer.subspan %[[SWITCH_RESULT]]#0[%[[SWITCH_RESULT]]#2] : !util.buffer{%[[SWITCH_RESULT]]#1} -> !util.buffer{%[[SWITCH_RESULT]]#3}
  // CHECK: util.return %[[SWITCH_RESULT]]#0, %[[SWITCH_RESULT]]#1, %[[SWITCH_RESULT]]#2, %[[SWITCH_RESULT]]#3
  util.return %0 : !util.buffer
}

// -----

// Tests that nested SCF operations propagate subranges correctly.

// CHECK-LABEL: @nestedScfOps
// CHECK-SAME: (%[[COND:.+]]: i1, %[[ARG:.+]]: !util.buffer, %[[ARG_SIZE:.+]]: index, %[[ARG_OFFSET:.+]]: index, %[[ARG_LENGTH:.+]]: index)
util.func private @nestedScfOps(%cond: i1, %arg: !util.buffer) -> !util.buffer {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %c1 = arith.constant 1 : index

  // CHECK-DAG: %[[ARG_SUBRANGE:.+]] = util.buffer.subspan %[[ARG]][%[[ARG_OFFSET]]] : !util.buffer{%[[ARG_SIZE]]} -> !util.buffer{%[[ARG_LENGTH]]}
  // CHECK-DAG: %{{.+}} = arith.constant 0
  // CHECK-DAG: %{{.+}} = arith.constant 5
  // CHECK-DAG: %{{.+}} = arith.constant 1

  // CHECK: %[[IF_RESULT:.+]]:4 = scf.if %[[COND]] -> (!util.buffer, index, index, index) {
  %0 = scf.if %cond -> !util.buffer {
    // CHECK-NEXT: %[[FOR_RESULT:.+]]:4 = scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}) -> (!util.buffer, index, index, index)
    %1 = scf.for %i = %c0 to %c5 step %c1 iter_args(%iter = %arg) -> !util.buffer {
      // CHECK-NEXT: %{{.+}} = util.buffer.subspan
      // CHECK-NEXT: scf.yield
      scf.yield %iter : !util.buffer
    }
    // CHECK: %{{.+}} = util.buffer.subspan %[[FOR_RESULT]]#0[%[[FOR_RESULT]]#2] : !util.buffer{%[[FOR_RESULT]]#1} -> !util.buffer{%[[FOR_RESULT]]#3}
    // CHECK-NEXT: scf.yield %[[FOR_RESULT]]#0, %[[FOR_RESULT]]#1, %[[FOR_RESULT]]#2, %[[FOR_RESULT]]#3
    scf.yield %1 : !util.buffer
  } else {
    // CHECK: scf.yield %[[ARG]], %[[ARG_SIZE]], %[[ARG_OFFSET]], %[[ARG_LENGTH]]
    scf.yield %arg : !util.buffer
  }
  // CHECK: %[[RESULT_SUBRANGE:.+]] = util.buffer.subspan %[[IF_RESULT]]#0[%[[IF_RESULT]]#2] : !util.buffer{%[[IF_RESULT]]#1} -> !util.buffer{%[[IF_RESULT]]#3}
  // CHECK: util.return %[[IF_RESULT]]#0, %[[IF_RESULT]]#1, %[[IF_RESULT]]#2, %[[IF_RESULT]]#3
  util.return %0 : !util.buffer
}

// -----

// Tests that function calls within scf ops get expanded as expected.
// This would also indicate that other constructs (global loads/etc) also get
// expanded within the SCF regions.

// CHECK-LABEL: @callerInSCF
// CHECK-SAME: (%[[RESOURCE:.+]]: !util.buffer, %[[STORAGE_SIZE:.+]]: index, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index, %[[COND:.+]]: i1)
util.func private @callerInSCF(%resource: !util.buffer, %cond: i1) {
  // NOTE: there will be extra stuff here from the arg insertion. The call
  // consumes the subranges and we expect the args to be passed directly.

  // CHECK: scf.if %[[COND]]
  scf.if %cond {
    // CHECK: util.call @callee(%[[RESOURCE]], %[[STORAGE_SIZE]], %[[OFFSET]], %[[LENGTH]])
    util.call @callee(%resource) : (!util.buffer) -> ()
  }

  util.return
}

util.func private @callee(%arg0: !util.buffer) {
  util.return
}

// -----

// Tests that existing subrange ops are used when propagating the ranges.
// This commonly shows up when there's a function call that passes in a subrange
// of a function argument: if we don't propagate the existing subranges then
// we cannot remove the subranges.

// CHECK-LABEL: @callerWithSubrange
// CHECK-SAME: (%[[ARG_RESOURCE:.+]]: !util.buffer, %[[ARG_SIZE:.+]]: index, %[[ARG_OFFSET:.+]]: index, %[[ARG_LENGTH:.+]]: index)
util.func private @callerWithSubrange(%arg: !util.buffer) {
  // NOTE: there will be extra stuff here from the arg insertion. The call
  // consumes the subranges and we expect the args to be passed directly.

  %arg_size = util.buffer.size %arg : !util.buffer
  // CHECK-DAG: %[[ARG_LOCAL_OFFSET:.+]] = arith.constant 100
  %arg_offset = arith.constant 100 : index
  // CHECK-DAG: %[[ARG_LOCAL_LENGTH:.+]] = arith.constant 200
  %arg_length = arith.constant 200 : index
  // CHECK-DAG: %[[ARG_ADJUSTED_OFFSET:.+]] = arith.addi %[[ARG_OFFSET]], %[[ARG_LOCAL_OFFSET]]
  %arg_subspan = util.buffer.subspan %arg[%arg_offset] : !util.buffer{%arg_size} -> !util.buffer{%arg_length}

  // CHECK: %[[RET0:.+]]:4 = util.call @callee(%[[ARG_RESOURCE]], %[[ARG_SIZE]], %[[ARG_ADJUSTED_OFFSET]], %[[ARG_LOCAL_LENGTH]])
  %ret0 = util.call @callee(%arg_subspan) : (!util.buffer) -> (!util.buffer)

  %ret0_size = util.buffer.size %ret0 : !util.buffer
  // CHECK-DAG: %[[RET0_LOCAL_OFFSET:.+]] = arith.constant 300
  %ret0_offset = arith.constant 300 : index
  // CHECK-DAG: %[[RET0_LOCAL_LENGTH:.+]] = arith.constant 400
  %ret0_length = arith.constant 400 : index
  // CHECK-DAG: %[[RET0_ADJUSTED_OFFSET:.+]] = arith.addi %[[RET0]]#2, %[[RET0_LOCAL_OFFSET]]
  %ret0_subspan = util.buffer.subspan %ret0[%ret0_offset] : !util.buffer{%ret0_size} -> !util.buffer{%ret0_length}

  // CHECK: %[[RET1:.+]]:4 = util.call @callee(%[[RET0]]#0, %[[RET0]]#1, %[[RET0_ADJUSTED_OFFSET]], %[[RET0_LOCAL_LENGTH]])
  %ret1 = util.call @callee(%ret0_subspan) : (!util.buffer) -> (!util.buffer)
  // CHECK: %[[RET1_SUBRANGE:.+]] = util.buffer.subspan %[[RET1]]#0[%[[RET1]]#2] : !util.buffer{%[[RET1]]#1} -> !util.buffer{%[[RET1]]#3}

  // CHECK-NEXT: util.optimization_barrier %[[RET1_SUBRANGE]] : !util.buffer
  util.optimization_barrier %ret1 : !util.buffer

  util.return
}

util.func private @callee(%arg0: !util.buffer) -> !util.buffer {
  util.return %arg0 : !util.buffer
}

// -----

// Tests that branch arguments are expanded into an explicit subrange of
// (resource, size, offset, length).
//
// This rotates subranges on branch operands into successors.

// CHECK-LABEL: @br
// CHECK-SAME: (%[[RESOURCE0:.+]]: !util.buffer, %[[STORAGE_SIZE0:.+]]: index, %[[OFFSET0:.+]]: index, %[[LENGTH0:.+]]: index, %[[RESOURCE1:.+]]: !util.buffer, %[[STORAGE_SIZE1:.+]]: index, %[[OFFSET1:.+]]: index, %[[LENGTH1:.+]]: index)
util.func private @br(%resource0: !util.buffer, %resource1: !util.buffer) {
  // NOTE: there will be extra stuff here from the arg insertion. The branch
  // consumes the unready resources and we expect the args to be passed directly
  // to the cf.br.

  // CHECK: cf.br ^bb1(%[[RESOURCE0]], %[[STORAGE_SIZE0]], %[[OFFSET0]], %[[LENGTH0]], %[[RESOURCE1]], %[[STORAGE_SIZE1]], %[[OFFSET1]], %[[LENGTH1]] : !util.buffer, index, index, index, !util.buffer, index, index, index)
  cf.br ^bb1(%resource0, %resource1 : !util.buffer, !util.buffer)

// CHECK-NEXT: ^bb1(%[[BB1_RESOURCE0:.+]]: !util.buffer, %[[BB1_STORAGE_SIZE0:.+]]: index, %[[BB1_OFFSET0:.+]]: index, %[[BB1_LENGTH0:.+]]: index, %[[BB1_RESOURCE1:.+]]: !util.buffer, %[[BB1_STORAGE_SIZE1:.+]]: index, %[[BB1_OFFSET1:.+]]: index, %[[BB1_LENGTH1:.+]]: index):
^bb1(%bb1_resource0: !util.buffer, %bb1_resource1: !util.buffer):
  // CHECK-NEXT: %[[BB1_SUBRANGE0:.+]] = util.buffer.subspan %[[BB1_RESOURCE0]][%[[BB1_OFFSET0]]] : !util.buffer{%[[BB1_STORAGE_SIZE0]]} -> !util.buffer{%[[BB1_LENGTH0]]}
  // CHECK-NEXT: %[[BB1_SUBRANGE1:.+]] = util.buffer.subspan %[[BB1_RESOURCE1]][%[[BB1_OFFSET1]]] : !util.buffer{%[[BB1_STORAGE_SIZE1]]} -> !util.buffer{%[[BB1_LENGTH1]]}

  // CHECK-NEXT: util.optimization_barrier %[[BB1_SUBRANGE0]]
  util.optimization_barrier %bb1_resource0 : !util.buffer
  // CHECK-NEXT: util.optimization_barrier %[[BB1_SUBRANGE1]]
  util.optimization_barrier %bb1_resource1 : !util.buffer

  util.return
}
