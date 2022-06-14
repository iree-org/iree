// RUN: iree-opt --split-input-file --iree-stream-propagate-subviews %s | FileCheck %s

// Tests that resource global loads also load all the subview params.
//
// This rotates subviews through stores and into loads.

//      CHECK: util.global private mutable @constantGlobal : !stream.resource<constant>
// CHECK-NEXT: util.global private mutable @constantGlobal__storage_size : index
// CHECK-NEXT: util.global private mutable @constantGlobal__offset : index
// CHECK-NEXT: util.global private mutable @constantGlobal__length : index
util.global private mutable @constantGlobal : !stream.resource<constant>

// CHECK-LABEL: @globalLoad
func.func @globalLoad() {
  // CHECK-NEXT: %[[RESOURCE:.+]] = util.global.load @constantGlobal : !stream.resource<constant>
  // CHECK-NEXT: %[[STORAGE_SIZE:.+]] = util.global.load @constantGlobal__storage_size : index
  // CHECK-NEXT: %[[OFFSET:.+]] = util.global.load @constantGlobal__offset : index
  // CHECK-NEXT: %[[LENGTH:.+]] = util.global.load @constantGlobal__length : index
  // CHECK: %[[SUBVIEW:.+]] = stream.resource.subview %[[RESOURCE]][%[[OFFSET]]] : !stream.resource<constant>{%[[STORAGE_SIZE]]} -> !stream.resource<constant>{%[[LENGTH]]}
  %0 = util.global.load @constantGlobal : !stream.resource<constant>
  // CHECK-NEXT: util.do_not_optimize(%[[SUBVIEW]])
  util.do_not_optimize(%0) : !stream.resource<constant>
  return
}

// -----

// Tests that resource global stores consume their incoming subviews.
//
// This rotates subviews through stores and into loads.

//      CHECK: util.global private mutable @mutableGlobal : !stream.resource<variable>
// CHECK-NEXT: util.global private mutable @mutableGlobal__storage_size : index
// CHECK-NEXT: util.global private mutable @mutableGlobal__offset : index
// CHECK-NEXT: util.global private mutable @mutableGlobal__length : index
util.global private mutable @mutableGlobal : !stream.resource<variable>

// CHECK-LABEL: @globalStore
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<variable>, %[[STORAGE_SIZE:.+]]: index, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index)
func.func @globalStore(%resource: !stream.resource<variable>) {
  // CHECK: util.global.store %[[RESOURCE]], @mutableGlobal : !stream.resource<variable>
  // CHECK: util.global.store %[[STORAGE_SIZE]], @mutableGlobal__storage_size : index
  // CHECK: util.global.store %[[OFFSET]], @mutableGlobal__offset : index
  // CHECK: util.global.store %[[LENGTH]], @mutableGlobal__length : index
  util.global.store %resource, @mutableGlobal : !stream.resource<variable>
  return
}

// -----

// Tests that function arguments are expanded into an explicit subview of
// (resource, size, offset, length).
//
// This rotates subviews from callers into callees.

// CHECK-LABEL: @funcArgs
// CHECK-SAME: (%[[RESOURCE0:.+]]: !stream.resource<external>, %[[STORAGE_SIZE0:.+]]: index, %[[OFFSET0:.+]]: index, %[[LENGTH0:.+]]: index, %[[RESOURCE1:.+]]: !stream.resource<transient>, %[[STORAGE_SIZE1:.+]]: index, %[[OFFSET1:.+]]: index, %[[LENGTH1:.+]]: index)
func.func @funcArgs(%resource0: !stream.resource<external>, %resource1: !stream.resource<transient>) {
  // CHECK-NEXT: %[[SUBVIEW0:.+]] = stream.resource.subview %[[RESOURCE0]][%[[OFFSET0]]] : !stream.resource<external>{%[[STORAGE_SIZE0]]} -> !stream.resource<external>{%[[LENGTH0]]}
  // CHECK-NEXT: %[[SUBVIEW1:.+]] = stream.resource.subview %[[RESOURCE1]][%[[OFFSET1]]] : !stream.resource<transient>{%[[STORAGE_SIZE1]]} -> !stream.resource<transient>{%[[LENGTH1]]}

  // CHECK-NEXT: util.do_not_optimize(%[[SUBVIEW0]])
  util.do_not_optimize(%resource0) : !stream.resource<external>
  // CHECK-NEXT: util.do_not_optimize(%[[SUBVIEW1]])
  util.do_not_optimize(%resource1) : !stream.resource<transient>
  return
}

// -----

// Tests that function results are expanded into an explicit subview of
// (resource, size, offset, length).
//
// This rotates subviews from callees into callers.

// CHECK-LABEL: @funcResults
// CHECK-SAME: (%[[RESOURCE0:.+]]: !stream.resource<external>, %[[STORAGE_SIZE0:.+]]: index, %[[OFFSET0:.+]]: index, %[[LENGTH0:.+]]: index, %[[RESOURCE1:.+]]: !stream.resource<transient>, %[[STORAGE_SIZE1:.+]]: index, %[[OFFSET1:.+]]: index, %[[LENGTH1:.+]]: index)
// CHECK-SAME: -> (!stream.resource<external>, index, index, index, !stream.resource<transient>, index, index, index)
func.func @funcResults(%resource0: !stream.resource<external>, %resource1: !stream.resource<transient>) -> (!stream.resource<external>, !stream.resource<transient>) {
  // NOTE: there will be extra stuff here from the arg insertion. Since the
  // return should consume the subview that was inserted we expect to directly
  // use the function arguments.

  // CHECK: return %[[RESOURCE0]], %[[STORAGE_SIZE0]], %[[OFFSET0]], %[[LENGTH0]], %[[RESOURCE1]], %[[STORAGE_SIZE1]], %[[OFFSET1]], %[[LENGTH1]]
  return %resource0, %resource1 : !stream.resource<external>, !stream.resource<transient>
}

// -----

// Tests that function calls have their args and results expanded into
// (resource, size, offset, length).
//
// This rotates subviews on args from callers to callees and subviews on results
// from callees to callers.

// CHECK-LABEL: @caller
// CHECK-SAME: (%[[RESOURCE0:.+]]: !stream.resource<external>, %[[STORAGE_SIZE0:.+]]: index, %[[OFFSET0:.+]]: index, %[[LENGTH0:.+]]: index, %[[RESOURCE1:.+]]: !stream.resource<transient>, %[[STORAGE_SIZE1:.+]]: index, %[[OFFSET1:.+]]: index, %[[LENGTH1:.+]]: index)
func.func @caller(%resource0: !stream.resource<external>, %resource1: !stream.resource<transient>) {
  // NOTE: there will be extra stuff here from the arg insertion. The call
  // consumes the subviews and we expect the args to be passed directly.

  // CHECK: %[[RET:.+]]:8 = call @callee(%[[RESOURCE0]], %[[STORAGE_SIZE0]], %[[OFFSET0]], %[[LENGTH0]],
  // CHECK-SAME:                         %[[RESOURCE1]], %[[STORAGE_SIZE1]], %[[OFFSET1]], %[[LENGTH1]])
  // CHECK-SAME: : (!stream.resource<external>, index, index, index, !stream.resource<transient>, index, index, index)
  // CHECK-SAME: -> (!stream.resource<external>, index, index, index, !stream.resource<transient>, index, index, index)
  %0:2 = call @callee(%resource0, %resource1) : (!stream.resource<external>, !stream.resource<transient>) -> (!stream.resource<external>, !stream.resource<transient>)
  // CHECK-NEXT: %[[RET_SUBVIEW0:.+]] = stream.resource.subview %[[RET]]#0[%[[RET]]#2] : !stream.resource<external>{%[[RET]]#1} -> !stream.resource<external>{%[[RET]]#3}
  // CHECK-NEXT: %[[RET_SUBVIEW1:.+]] = stream.resource.subview %[[RET]]#4[%[[RET]]#6] : !stream.resource<transient>{%[[RET]]#5} -> !stream.resource<transient>{%[[RET]]#7}

  // CHECK-NEXT: util.do_not_optimize(%[[RET_SUBVIEW0]]) : !stream.resource<external>
  util.do_not_optimize(%0#0) : !stream.resource<external>
  // CHECK-NEXT: util.do_not_optimize(%[[RET_SUBVIEW1]]) : !stream.resource<transient>
  util.do_not_optimize(%0#1) : !stream.resource<transient>

  return
}

func.func private @callee(%arg0: !stream.resource<external>, %arg1: !stream.resource<transient>) -> (!stream.resource<external>, !stream.resource<transient>)

// -----

// Tests that branch arguments are expanded into an explicit subview of
// (resource, size, offset, length).
//
// This rotates subviews on branch operands into successors.

// CHECK-LABEL: @br
// CHECK-SAME: (%[[RESOURCE0:.+]]: !stream.resource<external>, %[[STORAGE_SIZE0:.+]]: index, %[[OFFSET0:.+]]: index, %[[LENGTH0:.+]]: index, %[[RESOURCE1:.+]]: !stream.resource<transient>, %[[STORAGE_SIZE1:.+]]: index, %[[OFFSET1:.+]]: index, %[[LENGTH1:.+]]: index)
func.func @br(%resource0: !stream.resource<external>, %resource1: !stream.resource<transient>) {
  // NOTE: there will be extra stuff here from the arg insertion. The branch
  // consumes the unready resources and we expect the args to be passed directly
  // to the cf.br.

  // CHECK: cf.br ^bb1(%[[RESOURCE0]], %[[STORAGE_SIZE0]], %[[OFFSET0]], %[[LENGTH0]],
  // CHECK-SAME:    %[[RESOURCE1]], %[[STORAGE_SIZE1]], %[[OFFSET1]], %[[LENGTH1]] :
  cf.br ^bb1(%resource0, %resource1 : !stream.resource<external>, !stream.resource<transient>)

// CHECK-NEXT: ^bb1(%[[BB1_RESOURCE0:.+]]: !stream.resource<external>, %[[BB1_STORAGE_SIZE0:.+]]: index, %[[BB1_OFFSET0:.+]]: index, %[[BB1_LENGTH0:.+]]: index, %[[BB1_RESOURCE1:.+]]: !stream.resource<transient>, %[[BB1_STORAGE_SIZE1:.+]]: index, %[[BB1_OFFSET1:.+]]: index, %[[BB1_LENGTH1:.+]]: index):
^bb1(%bb1_resource0: !stream.resource<external>, %bb1_resource1: !stream.resource<transient>):
  // CHECK-NEXT: %[[BB1_SUBVIEW0:.+]] = stream.resource.subview %[[BB1_RESOURCE0]][%[[BB1_OFFSET0]]] : !stream.resource<external>{%[[BB1_STORAGE_SIZE0]]} -> !stream.resource<external>{%[[BB1_LENGTH0]]}
  // CHECK-NEXT: %[[BB1_SUBVIEW1:.+]] = stream.resource.subview %[[BB1_RESOURCE1]][%[[BB1_OFFSET1]]] : !stream.resource<transient>{%[[BB1_STORAGE_SIZE1]]} -> !stream.resource<transient>{%[[BB1_LENGTH1]]}

  // CHECK-NEXT: util.do_not_optimize(%[[BB1_SUBVIEW0]])
  util.do_not_optimize(%bb1_resource0) : !stream.resource<external>
  // CHECK-NEXT: util.do_not_optimize(%[[BB1_SUBVIEW1]])
  util.do_not_optimize(%bb1_resource1) : !stream.resource<transient>

  return
}
