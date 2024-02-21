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

  // CHECK: %[[RET:.+]]:8 = util.call @callee(%[[RESOURCE0]], %[[STORAGE_SIZE0]], %[[OFFSET0]], %[[LENGTH0]],
  // CHECK-SAME:                         %[[RESOURCE1]], %[[STORAGE_SIZE1]], %[[OFFSET1]], %[[LENGTH1]])
  // CHECK-SAME: : (!util.buffer, index, index, index, !util.buffer, index, index, index)
  // CHECK-SAME: -> (!util.buffer, index, index, index, !util.buffer, index, index, index)
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

  // CHECK: cf.br ^bb1(%[[RESOURCE0]], %[[STORAGE_SIZE0]], %[[OFFSET0]], %[[LENGTH0]],
  // CHECK-SAME:    %[[RESOURCE1]], %[[STORAGE_SIZE1]], %[[OFFSET1]], %[[LENGTH1]] :
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
