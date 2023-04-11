// RUN: iree-opt --split-input-file --iree-stream-schedule-allocation %s | FileCheck %s

// Tests that async constant ops get extracted into a dedicated constant op
// outside of the execution region. This allows us to handle them in various
// target-specific ways (such as using staging upload buffers if needed).

// CHECK-LABEL: @extractConstants
// CHECK-SAME: (%[[OPERAND_TIMEPOINT:.+]]: !stream.timepoint,
// CHECK-SAME:  %[[OPERAND:.+]]: !stream.resource<transient>,
// CHECK-SAME   %[[SIZE:.+]]: index)
func.func @extractConstants(%timepoint: !stream.timepoint, %operand: !stream.resource<transient>, %size: index) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c24 = arith.constant 24 : index
  %c128 = arith.constant 128 : index
  %c255_i32 = arith.constant 255 : i32

  // Constants get hoisted into a dedicated op.
  // CHECK: %[[CST_RETS:.+]]:2, %[[CST_TIMEPOINT:.+]] = stream.resource.constants :
  // CHECK-NEXT: !stream.resource<constant>{%c8} = dense<3> : tensor<8xi8>,
  // CHECK-NEXT: !stream.resource<constant>{%c16} = dense<4> : tensor<4x2xi16>

  // Remaining ops run in a normal execution region.
  // CHECK: %[[EXEC_TIMEPOINT:.+]] = stream.cmd.execute await(%[[OPERAND_TIMEPOINT]])
  // CHECK-SAME: => with(%[[OPERAND]]
  // CHECK-NEXT: stream.cmd.fill

  %results:3, %result_timepoint = stream.async.execute await(%timepoint) => with(%operand as %capture: !stream.resource<transient>{%size}) -> (!stream.resource<constant>{%c8}, !stream.resource<constant>{%c16}, !stream.resource<transient>{%size}) {
    %0 = stream.async.constant : !stream.resource<constant>{%c8} = dense<3> : tensor<8xi8>
    %1 = stream.async.constant : !stream.resource<constant>{%c16} = dense<4> : tensor<4x2xi16>
    %2 = stream.async.fill %c255_i32, %capture[%c0 to %c128 for %c128] : i32 -> %capture as !stream.resource<transient>{%size}
    stream.yield %0, %1, %2 : !stream.resource<constant>{%c8}, !stream.resource<constant>{%c16}, !stream.resource<transient>{%size}
  } => !stream.timepoint

  // Join the two async ops (constant upload and execution should overlap).
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[CST_TIMEPOINT]], %[[EXEC_TIMEPOINT]])
  // CHECK: util.optimization_barrier %[[JOIN]] : !stream.timepoint
  util.optimization_barrier %result_timepoint : !stream.timepoint

  // CHECK: util.optimization_barrier %[[CST_RETS]]#0
  util.optimization_barrier %results#0 : !stream.resource<constant>
  // CHECK: util.optimization_barrier %[[CST_RETS]]#1
  util.optimization_barrier %results#1 : !stream.resource<constant>
  // CHECK: util.optimization_barrier %[[OPERAND]]
  util.optimization_barrier %results#2 : !stream.resource<transient>
  return
}

// -----

// Tests that explicit allocations are preserved.

// CHECK-LABEL: @explicitAllocs
// CHECK-SAME: (%[[SIZE:.+]]: index)
func.func @explicitAllocs(%size: index) {
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc : !stream.resource<external>{%[[SIZE]]}
  %alloc = stream.resource.alloc : !stream.resource<external>{%size}
  // CHECK: util.optimization_barrier %[[ALLOC]]
  util.optimization_barrier %alloc : !stream.resource<external>

  %c0 = arith.constant 0 : index
  // CHECK: %[[EMPTY:.+]] = stream.resource.alloc : !stream.resource<transient>{%c0}
  %empty = stream.resource.alloc : !stream.resource<transient>{%c0}
  // CHECK: util.optimization_barrier %[[EMPTY]]
  util.optimization_barrier %empty : !stream.resource<transient>
  return
}

// -----

// Tests operands that pass directly through to results.
// These should be canonicalized away but are still valid.

// CHECK-LABEL: @passthroughOperands
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
func.func @passthroughOperands(%operand: !stream.resource<transient>, %size: index) {
  // CHECK: = stream.cmd.execute with(%[[OPERAND]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %result, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> (%operand as !stream.resource<transient>{%size}) {
    stream.yield %capture : !stream.resource<transient>{%size}
  // CHECK-NEXT: } => !stream.timepoint
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[OPERAND]]
  util.optimization_barrier %result : !stream.resource<transient>
  return
}

// -----

// CHECK-LABEL: @capturedOperands
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
func.func @capturedOperands(%operand: !stream.resource<transient>, %size: index) {
  // CHECK: stream.cmd.execute
  // CHECK-SAME: => with(%[[OPERAND]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]}
  %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) {
    // CHECK-NEXT: stream.cmd.copy %[[CAPTURE]]
    %0 = stream.async.clone %capture : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size}
    stream.yield
  } => !stream.timepoint
  return
}

// -----

// Tests operands that are tied to results with intermediate operations.

// CHECK-LABEL: @tiedOperands
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
func.func @tiedOperands(%operand: !stream.resource<transient>, %size: index) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c255_i32 = arith.constant 255 : i32
  // CHECK: stream.cmd.execute with(%[[OPERAND]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %result, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> (!stream.resource<transient>{%size}) {
    // CHECK-NEXT: stream.cmd.fill %c255_i32, %[[CAPTURE]]
    %0 = stream.async.fill %c255_i32, %capture[%c0 to %c128 for %c128] : i32 -> %capture as !stream.resource<transient>{%size}
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[OPERAND]]
  util.optimization_barrier %result : !stream.resource<transient>
  return
}

// -----

// Tests results are allocated for external use.
// We expect them to be allocated with synchronous alloc ops.

// CHECK-LABEL: @producedResults
// CHECK-SAME: (%[[SIZE0:.+]]: index, %[[SIZE1:.+]]: index)
func.func @producedResults(%size0: index, %size1: index) {
  %c254_i32 = arith.constant 254 : i32
  %c255_i32 = arith.constant 255 : i32
  // CHECK: %[[ALLOC_RETS:.+]]:2 = stream.resource.alloc uninitialized : !stream.resource<transient>{%[[SIZE0]]}, !stream.resource<transient>{%[[SIZE1]]}
  // CHECK: %[[TIMEPOINT:.+]] = stream.cmd.execute
  // CHECK-SAME: with(%[[ALLOC_RETS]]#0 as %[[CAPTURE0:.+]]: !stream.resource<transient>{%[[SIZE0]]},
  // CHECK-SAME:      %[[ALLOC_RETS]]#1 as %[[CAPTURE1:.+]]: !stream.resource<transient>{%[[SIZE1]]})
  %results:2, %result_timepoint = stream.async.execute with() -> (!stream.resource<transient>{%size0}, !stream.resource<transient>{%size1}) {
    // CHECK: stream.cmd.fill %c254_i32, %[[CAPTURE0]]
    %0 = stream.async.splat %c254_i32 : i32 -> !stream.resource<transient>{%size0}
    // CHECK: stream.cmd.fill %c255_i32, %[[CAPTURE1]]
    %1 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%size1}
    stream.yield %0, %1 : !stream.resource<transient>{%size0}, !stream.resource<transient>{%size1}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[TIMEPOINT]]
  util.optimization_barrier %result_timepoint : !stream.timepoint
  // CHECK: util.optimization_barrier %[[ALLOC_RETS]]#0
  util.optimization_barrier %results#0 : !stream.resource<transient>
  // CHECK: util.optimization_barrier %[[ALLOC_RETS]]#1
  util.optimization_barrier %results#1 : !stream.resource<transient>
  return
}

// -----

// Tests local values that are produced and consumed exclusively within the
// execution region. We expect them to be placed into packed slices and
// allocated with the async stream-ordered alloca/dealloca ops.

// CHECK-LABEL: @locals
// CHECK-SAME: (%[[SIZE0:.+]]: index, %[[SIZE1:.+]]: index, %[[AWAIT_TIMEPOINT:.+]]: !stream.timepoint)
func.func @locals(%size0: index, %size1: index, %await_timepoint: !stream.timepoint) -> !stream.timepoint {
  %c254_i32 = arith.constant 254 : i32
  %c255_i32 = arith.constant 255 : i32
  //      CHECK: %[[SLICES:.+]]:3 = stream.resource.pack on(#hal.affinity.queue<[0]>) slices({
  // CHECK-NEXT:   [0, 0] = %[[SIZE0]],
  // CHECK-NEXT:   [1, 1] = %[[SIZE1]]
  // CHECK-NEXT: })
  // CHECK-NEXT: %[[ALLOCA:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca uninitialized on(#hal.affinity.queue<[0]>) await(%[[AWAIT_TIMEPOINT]]) => !stream.resource<transient>{%[[SLICES]]#0} => !stream.timepoint
  // CHECK-NEXT: %[[AWAIT_JOIN:.+]] = stream.timepoint.join max(%[[AWAIT_TIMEPOINT]], %[[ALLOCA_TIMEPOINT]])
  // CHECK: %[[EXEC_TIMEPOINT:.+]] = stream.cmd.execute on(#hal.affinity.queue<[0]>) await(%[[AWAIT_JOIN]])
  // CHECK-SAME: with(%[[ALLOCA]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[SLICES]]#0})
  %result_timepoint = stream.async.execute on(#hal.affinity.queue<[0]>) await(%await_timepoint) => with() {
    // CHECK: stream.cmd.fill %c254_i32, %[[CAPTURE]][%[[SLICES]]#1 for %[[SIZE0]]] : i32 -> !stream.resource<transient>{%[[SLICES]]#0}
    %0 = stream.async.splat %c254_i32 : i32 -> !stream.resource<transient>{%size0}
    // CHECK: stream.cmd.fill %c255_i32, %[[CAPTURE]][%[[SLICES]]#2 for %[[SIZE1]]] : i32 -> !stream.resource<transient>{%[[SLICES]]#0}
    %1 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%size1}
    stream.yield
  } => !stream.timepoint
  // CHECK: %[[DEALLOCA_TIMEPOINT:.+]] = stream.resource.dealloca on(#hal.affinity.queue<[0]>) await(%[[EXEC_TIMEPOINT]]) => %[[ALLOCA]] : !stream.resource<transient>{%[[SLICES]]#0} => !stream.timepoint
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[DEALLOCA_TIMEPOINT]], %[[EXEC_TIMEPOINT]]) => !stream.timepoint
  // CHECK: return %[[JOIN]]
  return %result_timepoint : !stream.timepoint
}

// -----

// Tests that concurrently executable regions don't introduce new allocations.
// They should effectively be no-ops with respect to allocation so this looks
// a lot like like the above tests.

// CHECK-LABEL: @concurrentRegions
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
func.func @concurrentRegions(%operand: !stream.resource<transient>, %size: index) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c254_i32 = arith.constant 254 : i32
  %c255_i32 = arith.constant 255 : i32
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc uninitialized : !stream.resource<transient>{%[[SIZE]]}
  // CHECK: stream.cmd.execute
  // CHECK-SAME: with(%[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOC]] as %[[ALLOC_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %results:2, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) {
    // CHECK: stream.cmd.concurrent
    %0:2 = stream.async.concurrent with(%capture as %concurrent_capture: !stream.resource<transient>{%size}) -> (%capture as !stream.resource<transient>{%size}, !stream.resource<transient>{%size}) {
      // CHECK-NEXT: stream.cmd.fill %c254_i32, %[[OPERAND_CAPTURE]]
      %1 = stream.async.fill %c254_i32, %concurrent_capture[%c0 to %c128 for %c128] : i32 -> %concurrent_capture as !stream.resource<transient>{%size}
      // CHECK-NEXT: stream.cmd.fill %c255_i32, %[[ALLOC_CAPTURE]]
      %2 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%size}
      stream.yield %1, %2 : !stream.resource<transient>{%size}, !stream.resource<transient>{%size}
    }
    stream.yield %0#0, %0#1 : !stream.resource<transient>{%size}, !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[OPERAND]]
  util.optimization_barrier %results#0 : !stream.resource<transient>
  // CHECK: util.optimization_barrier %[[ALLOC]]
  util.optimization_barrier %results#1 : !stream.resource<transient>
  return
}

// -----

// CHECK-LABEL: @applyAsyncSplatOp
// CHECK-SAME: (%[[SIZE:.+]]: index)
func.func @applyAsyncSplatOp(%size: index) {
  %c255_i32 = arith.constant 255 : i32
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc uninitialized : !stream.resource<transient>{%[[SIZE]]}
  // CHECK: stream.cmd.execute with(%[[ALLOC]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %result, %result_timepoint = stream.async.execute with() -> (!stream.resource<transient>{%size}) {
    // CHECK: stream.cmd.fill %c255_i32, %[[CAPTURE]][%c0 for %[[SIZE]]] : i32 -> !stream.resource<transient>{%[[SIZE]]}
    %0 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%size}
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[ALLOC]]
  util.optimization_barrier %result : !stream.resource<transient>
  return
}

// -----

// CHECK-LABEL: @applyAsyncCloneOp
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
func.func @applyAsyncCloneOp(%operand: !stream.resource<transient>, %size: index) {
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc uninitialized : !stream.resource<transient>{%[[SIZE]]}
  // CHECK: stream.cmd.execute
  // CHECK-SAME: with(%[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOC]] as %[[ALLOC_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %result, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
    // CHECK: stream.cmd.copy %[[OPERAND_CAPTURE]][%c0], %[[ALLOC_CAPTURE]][%c0], %[[SIZE]]
    // CHECK-SAME: : !stream.resource<transient>{%[[SIZE]]} -> !stream.resource<transient>{%[[SIZE]]}
    %0 = stream.async.clone %capture : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size}
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[ALLOC]]
  util.optimization_barrier %result : !stream.resource<transient>
  return
}

// -----

// TODO(benvanik): place the allocation instead.
// NOTE: this should be placing the allocation but is currently a copy.

// CHECK-LABEL: @applyAsyncSliceOp
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
func.func @applyAsyncSliceOp(%operand: !stream.resource<transient>, %size: index) {
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %c144 = arith.constant 144 : index
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc uninitialized : !stream.resource<transient>{%c128}
  // CHECK: stream.cmd.execute
  // CHECK-SAME: with(%[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOC]] as %[[ALLOC_CAPTURE:.+]]: !stream.resource<transient>{%c128})
  %result, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%c128} {
    // CHECK: stream.cmd.copy %[[OPERAND_CAPTURE]][%c16], %[[ALLOC_CAPTURE]][%c0], %c128
    // CHECK-SAME: : !stream.resource<transient>{%[[SIZE]]} -> !stream.resource<transient>{%c128}
    %0 = stream.async.slice %capture[%c16 to %c144] : !stream.resource<transient>{%size} -> !stream.resource<transient>{%c128}
    stream.yield %0 : !stream.resource<transient>{%c128}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[ALLOC]]
  util.optimization_barrier %result : !stream.resource<transient>
  return
}

// -----

// CHECK-LABEL: @applyAsyncFillOp
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
func.func @applyAsyncFillOp(%operand: !stream.resource<transient>, %size: index) {
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %c144 = arith.constant 144 : index
  %c255_i32 = arith.constant 255 : i32
  // CHECK: stream.cmd.execute with(%[[OPERAND]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %result, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> (!stream.resource<transient>{%size}) {
    // CHECK: stream.cmd.fill %c255_i32, %[[CAPTURE]][%c16 for %c128] : i32 -> !stream.resource<transient>{%[[SIZE]]}
    %0 = stream.async.fill %c255_i32, %capture[%c16 to %c144 for %c128] : i32 -> %capture as !stream.resource<transient>{%size}
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[OPERAND]]
  util.optimization_barrier %result : !stream.resource<transient>
  return
}

// -----

// TODO(benvanik): place the allocation instead.
// NOTE: this should be placing the allocation but is currently a copy.

// CHECK-LABEL: @applyAsyncUpdateOp
// CHECK-SAME: (%[[UPDATE:.+]]: !stream.resource<external>,
// CHECK-SAME:  %[[OPERAND:.+]]: !stream.resource<transient>,
// CHECK-SAME:  %[[SIZE:.+]]: index)
func.func @applyAsyncUpdateOp(%update: !stream.resource<external>, %operand: !stream.resource<transient>, %size: index) {
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %c144 = arith.constant 144 : index
  // CHECK: stream.cmd.execute
  // CHECK-SAME: with(%[[UPDATE]] as %[[UPDATE_CAPTURE:.+]]: !stream.resource<external>{%c128},
  // CHECK-SAME:      %[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %result, %result_timepoint = stream.async.execute with(%update as %captured_update: !stream.resource<external>{%c128}, %operand as %captured_operand: !stream.resource<transient>{%size}) -> (!stream.resource<transient>{%size}) {
    // CHECK: stream.cmd.copy %[[UPDATE_CAPTURE]][%c0], %[[OPERAND_CAPTURE]][%c16], %c128
    // CHECK-SAME: : !stream.resource<external>{%c128} -> !stream.resource<transient>{%[[SIZE]]}
    %0 = stream.async.update %captured_update, %captured_operand[%c16 to %c144] : !stream.resource<external>{%c128} -> %captured_operand as !stream.resource<transient>{%size}
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[OPERAND]]
  util.optimization_barrier %result : !stream.resource<transient>
  return
}

// -----

// CHECK-LABEL: @applyAsyncCopyOp
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<external>,
// CHECK-SAME:  %[[TARGET:.+]]: !stream.resource<transient>,
// CHECK-SAME:  %[[SIZE:.+]]: index)
func.func @applyAsyncCopyOp(%source: !stream.resource<external>, %target: !stream.resource<transient>, %size: index) {
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %c144 = arith.constant 144 : index
  // CHECK: stream.cmd.execute
  // CHECK-SAME: with(%[[SOURCE]] as %[[SOURCE_CAPTURE:.+]]: !stream.resource<external>{%[[SIZE]]},
  // CHECK-SAME:      %[[TARGET]] as %[[TARGET_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %result, %result_timepoint = stream.async.execute with(%source as %captured_source: !stream.resource<external>{%size}, %target as %captured_target: !stream.resource<transient>{%size}) -> (!stream.resource<transient>{%size}) {
    // CHECK: stream.cmd.copy %[[SOURCE_CAPTURE]][%c16], %[[TARGET_CAPTURE]][%c16], %c128
    // CHECK-SAME: : !stream.resource<external>{%[[SIZE]]} -> !stream.resource<transient>{%[[SIZE]]}
    %0 = stream.async.copy %captured_source[%c16 to %c144], %captured_target[%c16 to %c144], %c128 : !stream.resource<external>{%size} -> %captured_operand as !stream.resource<transient>{%size}
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[TARGET]]
  util.optimization_barrier %result : !stream.resource<transient>
  return
}

// -----

// Tests that fully in-place execution regions with nesting don't allocate any
// transients. Both copies are concurrently in-place to the same provided
// target buffer.

// CHECK-LABEL: @applyConcurrentAsyncCopyOp
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<external>,
// CHECK-SAME:  %[[TARGET:.+]]: !stream.resource<transient>,
// CHECK-SAME:  %[[SIZE:.+]]: index)
func.func @applyConcurrentAsyncCopyOp(%source: !stream.resource<external>, %target: !stream.resource<transient>, %size: index) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %c144 = arith.constant 144 : index
  // CHECK: stream.cmd.execute
  // CHECK-SAME: with(%[[SOURCE]] as %[[SOURCE_CAPTURE:.+]]: !stream.resource<external>{%[[SIZE]]},
  // CHECK-SAME:      %[[TARGET]] as %[[TARGET_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %result, %result_timepoint = stream.async.execute with(%source as %captured_source: !stream.resource<external>{%size}, %target as %captured_target: !stream.resource<transient>{%size}) -> (%target as !stream.resource<transient>{%size}) {
    // CHECK: stream.cmd.concurrent
    %0 = stream.async.concurrent with(%captured_source as %concurrent_source: !stream.resource<external>{%size}, %captured_target as %concurrent_target: !stream.resource<transient>{%size}) -> (%captured_target as !stream.resource<transient>{%size}) {
      // CHECK: stream.cmd.copy %[[SOURCE_CAPTURE]][%c0], %[[TARGET_CAPTURE]][%c0], %c16
      // CHECK-SAME: : !stream.resource<external>{%[[SIZE]]} -> !stream.resource<transient>{%[[SIZE]]}
      %copy0 = stream.async.copy %concurrent_source[%c0 to %c16], %concurrent_target[%c0 to %c16], %c16 : !stream.resource<external>{%size} -> %concurrent_target as !stream.resource<transient>{%size}
      // CHECK: stream.cmd.copy %[[SOURCE_CAPTURE]][%c16], %[[TARGET_CAPTURE]][%c16], %c128
      // CHECK-SAME: : !stream.resource<external>{%[[SIZE]]} -> !stream.resource<transient>{%[[SIZE]]}
      %copy1 = stream.async.copy %concurrent_source[%c16 to %c144], %copy0[%c16 to %c144], %c128 : !stream.resource<external>{%size} -> %copy0 as !stream.resource<transient>{%size}
      stream.yield %copy1 : !stream.resource<transient>{%size}
    }
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[TARGET]]
  util.optimization_barrier %result : !stream.resource<transient>
  return
}

// -----

// TODO(#11249): add a test for in-place collectives (send == recv).

// CHECK-LABEL: @applyAsyncCollectiveOpOutOfPlace
// CHECK-SAME: (%[[SEND:.+]]: !stream.resource<external>, %[[SEND_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[RECV:.+]]: !stream.resource<transient>, %[[RECV_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[COUNT:[a-z0-9]+]]: index)
func.func @applyAsyncCollectiveOpOutOfPlace(%send: !stream.resource<external>, %send_size: index, %recv: !stream.resource<transient>, %recv_size: index, %count: index) {
  %c0 = arith.constant 0 : index
  %channel = stream.channel.default : !stream.channel
  // CHECK: stream.cmd.execute
  // CHECK-SAME: with(%[[SEND]] as %[[SEND_CAPTURE:.+]]: !stream.resource<external>{%[[SEND_SIZE]]},
  // CHECK-SAME:      %[[RECV]] as %[[RECV_CAPTURE:.+]]: !stream.resource<transient>{%[[RECV_SIZE]]})
  %result, %result_timepoint = stream.async.execute with(%send as %captured_send: !stream.resource<external>{%send_size}, %recv as %captured_recv: !stream.resource<transient>{%recv_size}) -> (!stream.resource<transient>{%recv_size}) {
    // CHECK: stream.cmd.collective<all_gather : f32>[%[[COUNT]]]
    %0 = stream.async.collective<all_gather : f32>[%count] channel(%channel)
        // CHECK-NEXT: ro %[[SEND_CAPTURE]][%c0 for %[[SEND_SIZE]]] : !stream.resource<external>{%[[SEND_SIZE]]}
        %captured_send[%c0 to %send_size for %send_size],
        // CHECK-NEXT: wo %[[RECV_CAPTURE]][%c0 for %[[RECV_SIZE]]] : !stream.resource<transient>{%[[RECV_SIZE]]}
        %captured_recv[%c0 to %recv_size for %recv_size] :
        !stream.resource<external>{%send_size} -> %captured_recv as !stream.resource<transient>{%recv_size}
    stream.yield %0 : !stream.resource<transient>{%recv_size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[RECV]]
  util.optimization_barrier %result : !stream.resource<transient>
  return
}

// -----

// TODO(benvanik): test affinity changes that would introduce invalidate/fill.

// CHECK-LABEL: @applyAsyncTransferOp
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
func.func @applyAsyncTransferOp(%operand: !stream.resource<transient>, %size: index) {
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc uninitialized : !stream.resource<transient>{%[[SIZE]]}
  // CHECK: stream.cmd.execute
  // CHECK-SAME: with(%[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOC]] as %[[ALLOC_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %result, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
    // CHECK: stream.cmd.copy %[[OPERAND_CAPTURE]][%c0], %[[ALLOC_CAPTURE]][%c0], %[[SIZE]]
    // CHECK-SAME: : !stream.resource<transient>{%[[SIZE]]} -> !stream.resource<transient>{%[[SIZE]]}
    %0 = stream.async.transfer %capture : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size}
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[ALLOC]]
  util.optimization_barrier %result : !stream.resource<transient>
  return
}

// -----

// CHECK-LABEL: @applyAsyncDispatchOp
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[OFFSET:.+]]: index, %[[END:.+]]: index, %[[LENGTH:.+]]: index)
func.func @applyAsyncDispatchOp(%operand: !stream.resource<transient>, %size: index, %offset: index, %end: index, %length: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc uninitialized : !stream.resource<transient>{%[[SIZE]]}
  // CHECK: %[[TIMEPOINT:.+]] = stream.cmd.execute
  // CHECK-SAME: with(%[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOC]] as %[[ALLOC_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %results:2, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> (%operand as !stream.resource<transient>{%size}, !stream.resource<transient>{%size}) {
    // CHECK-NEXT: stream.cmd.dispatch @executable::@dispatch[%c1, %c1, %c1](%c4 : index) {
    // CHECK-NEXT:   rw %[[OPERAND_CAPTURE]][%[[OFFSET]] for %[[LENGTH]]] : !stream.resource<transient>{%[[SIZE]]},
    // CHECK-NEXT:   wo %[[ALLOC_CAPTURE]][%c0{{[_0-9]*}} for %[[SIZE]]] : !stream.resource<transient>{%[[SIZE]]}
    // CHECK-NEXT: }
    %0:2 = stream.async.dispatch @executable::@dispatch[%c1, %c1, %c1](%capture[%offset to %end for %length], %c4) : (!stream.resource<transient>{%size}, index) -> (%capture{%size}, !stream.resource<transient>{%size})
    stream.yield %0#0, %0#1 : !stream.resource<transient>{%size}, !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[TIMEPOINT]]
  util.optimization_barrier %result_timepoint : !stream.timepoint
  // CHECK: util.optimization_barrier %[[OPERAND]]
  util.optimization_barrier %results#0 : !stream.resource<transient>
  // CHECK: util.optimization_barrier %[[ALLOC]]
  util.optimization_barrier %results#1 : !stream.resource<transient>
  return
}

// -----

// CHECK: stream.cmd.func private @asyncExtern(%arg0[%arg1 for %arg2]: !stream.resource<*>, %arg3: index, %arg4[%arg5 for %arg6]: !stream.resource<*>)
stream.async.func private @asyncExtern(%arg0: !stream.resource<*>, %arg1: index) -> (%arg0, !stream.resource<*>)

// CHECK-LABEL: @applyAsyncCallOp
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[OFFSET:.+]]: index, %[[END:.+]]: index, %[[LENGTH:.+]]: index)
func.func @applyAsyncCallOp(%operand: !stream.resource<transient>, %size: index, %offset: index, %end: index, %length: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc uninitialized : !stream.resource<transient>{%[[SIZE]]}
  // CHECK: %[[TIMEPOINT:.+]] = stream.cmd.execute
  // CHECK-SAME: with(%[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOC]] as %[[ALLOC_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %results:2, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> (%operand as !stream.resource<transient>{%size}, !stream.resource<transient>{%size}) {
    // CHECK-NEXT: stream.cmd.call @asyncExtern(rw %[[OPERAND_CAPTURE]][%[[OFFSET]] for %[[LENGTH]]], %c4, wo %[[ALLOC_CAPTURE]][%c0{{[_0-9]*}} for %[[SIZE]]]) :
    // CHECK-SAME:     (!stream.resource<transient>{%[[SIZE]]}, index, !stream.resource<transient>{%[[SIZE]]}) -> ()
    %0:2 = stream.async.call @asyncExtern(%capture[%offset to %end for %length], %c4) : (!stream.resource<transient>{%size}, index) -> (%capture{%size}, !stream.resource<transient>{%size})
    stream.yield %0#0, %0#1 : !stream.resource<transient>{%size}, !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[TIMEPOINT]]
  util.optimization_barrier %result_timepoint : !stream.timepoint
  // CHECK: util.optimization_barrier %[[OPERAND]]
  util.optimization_barrier %results#0 : !stream.resource<transient>
  // CHECK: util.optimization_barrier %[[ALLOC]]
  util.optimization_barrier %results#1 : !stream.resource<transient>
  return
}

// -----

// Tests that stream.async.load/store are converted to their explicit forms.

// CHECK-LABEL: @asyncLoadStore
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<staging>,
// CHECK-SAME:  %[[SIZE:.+]]: index)
func.func @asyncLoadStore(%operand: !stream.resource<staging>, %size: index) -> f32 {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 5.4 : f32
  // CHECK: stream.resource.store %cst, %[[OPERAND]][%c0] : f32 -> !stream.resource<staging>{%[[SIZE]]}
  %0 = stream.async.store %cst, %operand[%c0] : f32 -> %operand as !stream.resource<staging>{%size}
  // CHECK: %[[RESULT:.+]] = stream.resource.load %[[OPERAND]][%c0] : !stream.resource<staging>{%[[SIZE]]} -> f32
  %1 = stream.async.load %0[%c0] : !stream.resource<staging>{%size} -> f32
  // CHECK: return %[[RESULT]]
  return %1 : f32
}
