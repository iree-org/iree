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
func.func @globalLoad() {
  // CHECK-NEXT: %[[RESOURCE:.+]] = util.global.load @constantGlobal : !util.buffer
  // CHECK-NEXT: %[[STORAGE_SIZE:.+]] = util.global.load @constantGlobal__storage_size : index
  // CHECK-NEXT: %[[OFFSET:.+]] = util.global.load @constantGlobal__offset : index
  // CHECK-NEXT: %[[LENGTH:.+]] = util.global.load @constantGlobal__length : index
  // CHECK: %[[SUBRANGE:.+]] = util.buffer.subspan %[[RESOURCE]][%[[OFFSET]]] : !util.buffer{%[[STORAGE_SIZE]]} -> !util.buffer{%[[LENGTH]]}
  %0 = util.global.load @constantGlobal : !util.buffer
  // CHECK-NEXT: util.do_not_optimize(%[[SUBRANGE]])
  util.do_not_optimize(%0) : !util.buffer
  return
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
func.func @globalStore(%resource: !util.buffer) {
  // CHECK: util.global.store %[[RESOURCE]], @mutableGlobal : !util.buffer
  // CHECK: util.global.store %[[STORAGE_SIZE]], @mutableGlobal__storage_size : index
  // CHECK: util.global.store %[[OFFSET]], @mutableGlobal__offset : index
  // CHECK: util.global.store %[[LENGTH]], @mutableGlobal__length : index
  util.global.store %resource, @mutableGlobal : !util.buffer
  return
}

// -----

// Tests that function arguments are expanded into an explicit subrange of
// (resource, size, offset, length).
//
// This rotates subranges from callers into callees.

// CHECK-LABEL: @funcArgs
// CHECK-SAME: (%[[RESOURCE0:.+]]: !util.buffer, %[[STORAGE_SIZE0:.+]]: index, %[[OFFSET0:.+]]: index, %[[LENGTH0:.+]]: index, %[[RESOURCE1:.+]]: !util.buffer, %[[STORAGE_SIZE1:.+]]: index, %[[OFFSET1:.+]]: index, %[[LENGTH1:.+]]: index)
func.func @funcArgs(%resource0: !util.buffer, %resource1: !util.buffer) {
  // CHECK-NEXT: %[[SUBRANGE0:.+]] = util.buffer.subspan %[[RESOURCE0]][%[[OFFSET0]]] : !util.buffer{%[[STORAGE_SIZE0]]} -> !util.buffer{%[[LENGTH0]]}
  // CHECK-NEXT: %[[SUBRANGE1:.+]] = util.buffer.subspan %[[RESOURCE1]][%[[OFFSET1]]] : !util.buffer{%[[STORAGE_SIZE1]]} -> !util.buffer{%[[LENGTH1]]}

  // CHECK-NEXT: util.do_not_optimize(%[[SUBRANGE0]])
  util.do_not_optimize(%resource0) : !util.buffer
  // CHECK-NEXT: util.do_not_optimize(%[[SUBRANGE1]])
  util.do_not_optimize(%resource1) : !util.buffer
  return
}

// -----

// Tests that function results are expanded into an explicit subrange of
// (resource, size, offset, length).
//
// This rotates subranges from callees into callers.

// CHECK-LABEL: @funcResults
// CHECK-SAME: (%[[RESOURCE0:.+]]: !util.buffer, %[[STORAGE_SIZE0:.+]]: index, %[[OFFSET0:.+]]: index, %[[LENGTH0:.+]]: index, %[[RESOURCE1:.+]]: !util.buffer, %[[STORAGE_SIZE1:.+]]: index, %[[OFFSET1:.+]]: index, %[[LENGTH1:.+]]: index)
// CHECK-SAME: -> (!util.buffer, index, index, index, !util.buffer, index, index, index)
func.func @funcResults(%resource0: !util.buffer, %resource1: !util.buffer) -> (!util.buffer, !util.buffer) {
  // NOTE: there will be extra stuff here from the arg insertion. Since the
  // return should consume the subrange that was inserted we expect to directly
  // use the function arguments.

  // CHECK: return %[[RESOURCE0]], %[[STORAGE_SIZE0]], %[[OFFSET0]], %[[LENGTH0]], %[[RESOURCE1]], %[[STORAGE_SIZE1]], %[[OFFSET1]], %[[LENGTH1]]
  return %resource0, %resource1 : !util.buffer, !util.buffer
}

// -----

// Tests that function calls have their args and results expanded into
// (resource, size, offset, length).
//
// This rotates subranges on args from callers to callees and subranges on results
// from callees to callers.

// CHECK-LABEL: @caller
// CHECK-SAME: (%[[RESOURCE0:.+]]: !util.buffer, %[[STORAGE_SIZE0:.+]]: index, %[[OFFSET0:.+]]: index, %[[LENGTH0:.+]]: index, %[[RESOURCE1:.+]]: !util.buffer, %[[STORAGE_SIZE1:.+]]: index, %[[OFFSET1:.+]]: index, %[[LENGTH1:.+]]: index)
func.func @caller(%resource0: !util.buffer, %resource1: !util.buffer) {
  // NOTE: there will be extra stuff here from the arg insertion. The call
  // consumes the subranges and we expect the args to be passed directly.

  // CHECK: %[[RET:.+]]:8 = call @callee(%[[RESOURCE0]], %[[STORAGE_SIZE0]], %[[OFFSET0]], %[[LENGTH0]],
  // CHECK-SAME:                         %[[RESOURCE1]], %[[STORAGE_SIZE1]], %[[OFFSET1]], %[[LENGTH1]])
  // CHECK-SAME: : (!util.buffer, index, index, index, !util.buffer, index, index, index)
  // CHECK-SAME: -> (!util.buffer, index, index, index, !util.buffer, index, index, index)
  %0:2 = call @callee(%resource0, %resource1) : (!util.buffer, !util.buffer) -> (!util.buffer, !util.buffer)
  // CHECK-NEXT: %[[RET_SUBRANGE0:.+]] = util.buffer.subspan %[[RET]]#0[%[[RET]]#2] : !util.buffer{%[[RET]]#1} -> !util.buffer{%[[RET]]#3}
  // CHECK-NEXT: %[[RET_SUBRANGE1:.+]] = util.buffer.subspan %[[RET]]#4[%[[RET]]#6] : !util.buffer{%[[RET]]#5} -> !util.buffer{%[[RET]]#7}

  // CHECK-NEXT: util.do_not_optimize(%[[RET_SUBRANGE0]]) : !util.buffer
  util.do_not_optimize(%0#0) : !util.buffer
  // CHECK-NEXT: util.do_not_optimize(%[[RET_SUBRANGE1]]) : !util.buffer
  util.do_not_optimize(%0#1) : !util.buffer

  return
}

func.func private @callee(%arg0: !util.buffer, %arg1: !util.buffer) -> (!util.buffer, !util.buffer)

// -----

// Tests that branch arguments are expanded into an explicit subrange of
// (resource, size, offset, length).
//
// This rotates subranges on branch operands into successors.

// CHECK-LABEL: @br
// CHECK-SAME: (%[[RESOURCE0:.+]]: !util.buffer, %[[STORAGE_SIZE0:.+]]: index, %[[OFFSET0:.+]]: index, %[[LENGTH0:.+]]: index, %[[RESOURCE1:.+]]: !util.buffer, %[[STORAGE_SIZE1:.+]]: index, %[[OFFSET1:.+]]: index, %[[LENGTH1:.+]]: index)
func.func @br(%resource0: !util.buffer, %resource1: !util.buffer) {
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

  // CHECK-NEXT: util.do_not_optimize(%[[BB1_SUBRANGE0]])
  util.do_not_optimize(%bb1_resource0) : !util.buffer
  // CHECK-NEXT: util.do_not_optimize(%[[BB1_SUBRANGE1]])
  util.do_not_optimize(%bb1_resource1) : !util.buffer

  return
}
