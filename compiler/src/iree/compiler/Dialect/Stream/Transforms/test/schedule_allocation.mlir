// RUN: iree-opt --split-input-file --iree-stream-schedule-allocation %s | FileCheck %s

// Tests that async constant ops get extracted into a dedicated constant op
// outside of the execution region. This allows us to handle them in various
// target-specific ways (such as using staging upload buffers if needed).

// CHECK-LABEL: @extractConstants
// CHECK-SAME: (%[[OPERAND_TIMEPOINT:.+]]: !stream.timepoint,
// CHECK-SAME:  %[[OPERAND:.+]]: !stream.resource<transient>,
// CHECK-SAME   %[[SIZE:.+]]: index)
util.func public @extractConstants(%timepoint: !stream.timepoint, %operand: !stream.resource<transient>, %size: index) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c24 = arith.constant 24 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %c255_i32 = arith.constant 255 : i32

  // Constants get hoisted into a dedicated op.
  // CHECK: %[[CST_RETS:.+]]:2, %[[CST_TIMEPOINT:.+]] = stream.resource.constants :
  // CHECK-NEXT: !stream.resource<constant>{%c8} = dense<3> : tensor<8xi8>,
  // CHECK-NEXT: !stream.resource<constant>{%c16} = dense<4> : tensor<4x2xi16>

  // Initialized variables get hoisted into a dedicated op.
  // CHECK: %[[VAR_RET:.+]], %[[VAR_TIMEPOINT:.+]] = stream.resource.constants :
  // CHECK-NEXT: !stream.resource<variable>{%c32} = dense<5> : tensor<8xi32>

  // Remaining ops run in a normal execution region.
  // CHECK: %[[EXEC_TIMEPOINT:.+]] = stream.cmd.execute await(%[[OPERAND_TIMEPOINT]])
  // CHECK-SAME: => with(%[[OPERAND]]
  // CHECK-NEXT: stream.cmd.fill

  %results:4, %result_timepoint = stream.async.execute await(%timepoint) => with(%operand as %capture: !stream.resource<transient>{%size}) -> (!stream.resource<constant>{%c8}, !stream.resource<constant>{%c16}, !stream.resource<variable>{%c32}, !stream.resource<transient>{%size}) {
    %0 = stream.async.constant : !stream.resource<constant>{%c8} = dense<3> : tensor<8xi8>
    %1 = stream.async.constant : !stream.resource<constant>{%c16} = dense<4> : tensor<4x2xi16>
    %2 = stream.async.constant : !stream.resource<variable>{%c32} = dense<5> : tensor<8xi32>
    %3 = stream.async.fill %c255_i32, %capture[%c0 to %c128 for %c128] : i32 -> %capture as !stream.resource<transient>{%size}
    stream.yield %0, %1, %2, %3 : !stream.resource<constant>{%c8}, !stream.resource<constant>{%c16}, !stream.resource<variable>{%c32}, !stream.resource<transient>{%size}
  } => !stream.timepoint

  // Join the two async ops (constant upload and execution should overlap).
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[CST_TIMEPOINT]], %[[VAR_TIMEPOINT]], %[[EXEC_TIMEPOINT]])
  // CHECK: util.optimization_barrier %[[JOIN]] : !stream.timepoint
  util.optimization_barrier %result_timepoint : !stream.timepoint

  // CHECK: util.optimization_barrier %[[CST_RETS]]#0
  util.optimization_barrier %results#0 : !stream.resource<constant>
  // CHECK: util.optimization_barrier %[[CST_RETS]]#1
  util.optimization_barrier %results#1 : !stream.resource<constant>
  // CHECK: util.optimization_barrier %[[VAR_RET]]
  util.optimization_barrier %results#2 : !stream.resource<variable>
  // CHECK: util.optimization_barrier %[[OPERAND]]
  util.optimization_barrier %results#3 : !stream.resource<transient>
  util.return
}

// -----

// Tests that constants are extracted and bucketed by their usage affinity.
// Constants _should_ only be allocated for their users once extracted and only
// include the execution affinity of the original region if they are used within
// it. Today we don't track this properly and all constants extracted from an
// execution op are assumed to be accessed by it - if we could have identified
// them as being used exclusively on other affinities earlier we would not have
// put them into the same region on the wrong affinity.

// CHECK-LABEL: @extractConstantsMultiAffinity
// CHECK-SAME: (%[[WAIT_TIMEPOINT:.+]]: !stream.timepoint)
util.func public @extractConstantsMultiAffinity(%wait_timepoint: !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c255_i32 = arith.constant 255 : i32

  // <constant>{%c8} is produced on @device_a + used on @device_b:
  // CHECK: %[[CST8:.+]], %[[CST8_TIMEPOINT:.+]] = stream.resource.constants
  // CHECK-SAME: on(#hal.device.optimal<[#hal.device.promise<@device_a>, #hal.device.promise<@device_b>]>)
  // CHECK-NEXT: !stream.resource<constant>{%c8} = dense<3> : tensor<8xi8>

  // <constant>{%c16} is produced on @device_a + used on @device_c + @device_d:
  // CHECK: %[[CST16:.+]], %[[CST16_TIMEPOINT:.+]] = stream.resource.constants
  // CHECK-SAME: on(#hal.device.optimal<[#hal.device.promise<@device_a>, #hal.device.promise<@device_c>, #hal.device.promise<@device_d>]>)
  // CHECK-NEXT: !stream.resource<constant>{%c16} = dense<4> : tensor<4x2xi16>

  // <constant>{%c32} is produced on @device_a + mutated on @device_e:
  // CHECK: %[[CST32:.+]], %[[CST32_TIMEPOINT:.+]] = stream.resource.constants
  // CHECK-SAME: on(#hal.device.optimal<[#hal.device.promise<@device_a>, #hal.device.promise<@device_e>]>)
  // CHECK-NEXT: !stream.resource<variable>{%c32} = dense<5> : tensor<8xi32>

  // Join on wait and CST32 for execution, as the fill only needs to wait until
  // it is loaded (and not on CST8/CST16).
  // CHECK: %[[WAIT_FILL_TIMEPOINT:.+]] = stream.timepoint.join max(%[[WAIT_TIMEPOINT]], %[[CST32_TIMEPOINT]])

  // Fill runs on CST32 only.
  // CHECK: %[[FILL_TIMEPOINT:.+]] = stream.cmd.execute on(#hal.device.promise<@device_a>)
  // CHECK-SAME: await(%[[WAIT_FILL_TIMEPOINT]]) => with(%[[CST32]]
  // CHECK-NEXT: stream.cmd.fill

  %results:3, %result_timepoint = stream.async.execute on(#hal.device.promise<@device_a>) await(%wait_timepoint) => with() -> (!stream.resource<constant>{%c8}, !stream.resource<constant>{%c16}, !stream.resource<variable>{%c32}) {
    %cst8 = stream.async.constant : !stream.resource<constant>{%c8} = dense<3> : tensor<8xi8>
    %cst16 = stream.async.constant : !stream.resource<constant>{%c16} = dense<4> : tensor<4x2xi16>
    %cst32 = stream.async.constant : !stream.resource<variable>{%c32} = dense<5> : tensor<8xi32>
    %fill_cst32 = stream.async.fill %c255_i32, %cst32[%c0 to %c16 for %c16] : i32 -> %cst32 as !stream.resource<variable>{%c32}
    stream.yield %cst8, %cst16, %fill_cst32 : !stream.resource<constant>{%c8}, !stream.resource<constant>{%c16}, !stream.resource<variable>{%c32}
  } => !stream.timepoint

  // Join the async ops (constant uploads and execution should overlap).
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[CST8_TIMEPOINT]], %[[CST16_TIMEPOINT]], %[[FILL_TIMEPOINT]])
  // CHECK: util.optimization_barrier %[[JOIN]] : !stream.timepoint
  util.optimization_barrier %result_timepoint : !stream.timepoint

  // As if used on another device:
  // CHECK: stream.cmd.execute on(#hal.device.promise<@device_b>)
  // CHECK-SAME: await(%[[JOIN]]) => with(%[[CST8]]
  %result_b, %result_timepoint_b = stream.async.execute on(#hal.device.promise<@device_b>) await(%result_timepoint) => with(%results#0 as %capture: !stream.resource<constant>{%c8}) -> (!stream.resource<constant>{%c8}) {
    stream.yield %capture : !stream.resource<constant>{%c8}
  } => !stream.timepoint

  // As if used on multiple devices:
  // CHECK: stream.cmd.execute on(#hal.device.promise<@device_c>)
  // CHECK-SAME: await(%[[JOIN]]) => with(%[[CST16]]
  %result_c, %result_timepoint_c = stream.async.execute on(#hal.device.promise<@device_c>) await(%result_timepoint) => with(%results#1 as %capture: !stream.resource<constant>{%c16}) -> (!stream.resource<constant>{%c16}) {
    stream.yield %capture : !stream.resource<constant>{%c16}
  } => !stream.timepoint
  // CHECK: stream.cmd.execute on(#hal.device.promise<@device_d>)
  // CHECK-SAME: await(%[[JOIN]]) => with(%[[CST16]]
  %result_d, %result_timepoint_d = stream.async.execute on(#hal.device.promise<@device_d>) await(%result_timepoint) => with(%results#1 as %capture: !stream.resource<constant>{%c16}) -> (!stream.resource<constant>{%c16}) {
    stream.yield %capture : !stream.resource<constant>{%c16}
  } => !stream.timepoint

  // As if mutated on another device:
  // CHECK: stream.cmd.execute on(#hal.device.promise<@device_e>)
  // CHECK-SAME: await(%[[JOIN]]) => with(%[[CST32]]
  %result_e, %result_timepoint_e = stream.async.execute on(#hal.device.promise<@device_e>) await(%result_timepoint) => with(%results#2 as %capture: !stream.resource<variable>{%c32}) -> %results#2 as !stream.resource<variable>{%c32} {
    stream.yield %capture : !stream.resource<variable>{%c32}
  } => !stream.timepoint

  util.return
}

// -----

// Tests that execution regions in initializers are marked as `once` indicating
// that they are one-shot. The analysis today only checks for ops within the
// first block of an initializer and treats all others as reusable.

// CHECK-LABEL: util.initializer
util.initializer {
  %c254_i32 = arith.constant 254 : i32
  %size = arith.constant 128 : index
  // CHECK: = stream.cmd.execute once
  %result, %result_timepoint = stream.async.execute with() -> !stream.resource<transient>{%size} {
    %0 = stream.async.splat %c254_i32 : i32 -> !stream.resource<transient>{%size}
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.optimization_barrier %result : !stream.resource<transient>
  util.return
}

// -----

// Tests that explicit allocations are preserved.

// CHECK-LABEL: @explicitAllocs
// CHECK-SAME: (%[[SIZE:.+]]: index)
util.func public @explicitAllocs(%size: index) {
  // CHECK: %[[ALLOC:.+]] = stream.resource.alloc : !stream.resource<external>{%[[SIZE]]}
  %alloc = stream.resource.alloc : !stream.resource<external>{%size}
  // CHECK: util.optimization_barrier %[[ALLOC]]
  util.optimization_barrier %alloc : !stream.resource<external>

  %c0 = arith.constant 0 : index
  // CHECK: %[[EMPTY:.+]] = stream.resource.alloc : !stream.resource<transient>{%c0}
  %empty = stream.resource.alloc : !stream.resource<transient>{%c0}
  // CHECK: util.optimization_barrier %[[EMPTY]]
  util.optimization_barrier %empty : !stream.resource<transient>
  util.return
}

// -----

// Tests operands that pass directly through to results.
// These should be canonicalized away but are still valid.

// CHECK-LABEL: @passthroughOperands
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
util.func public @passthroughOperands(%operand: !stream.resource<transient>, %size: index) {
  // CHECK: = stream.cmd.execute with(%[[OPERAND]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %result, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> (%operand as !stream.resource<transient>{%size}) {
    stream.yield %capture : !stream.resource<transient>{%size}
  // CHECK-NEXT: } => !stream.timepoint
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[OPERAND]]
  util.optimization_barrier %result : !stream.resource<transient>
  util.return
}

// -----

// CHECK-LABEL: @capturedOperands
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
util.func public @capturedOperands(%operand: !stream.resource<transient>, %size: index) {
  // CHECK: stream.cmd.execute
  // CHECK-SAME: => with(%[[OPERAND]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]}
  %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) {
    // CHECK-NEXT: stream.cmd.copy %[[CAPTURE]]
    %0 = stream.async.clone %capture : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size}
    stream.yield
  } => !stream.timepoint
  util.return
}

// -----

// Tests operands that are tied to results with intermediate operations.

// CHECK-LABEL: @tiedOperands
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
util.func public @tiedOperands(%operand: !stream.resource<transient>, %size: index) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c255_i32 = arith.constant 255 : i32
  // CHECK: stream.cmd.execute with(%[[OPERAND]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %result, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> (%operand as !stream.resource<transient>{%size}) {
    // CHECK-NEXT: stream.cmd.fill %c255_i32, %[[CAPTURE]]
    %0 = stream.async.fill %c255_i32, %capture[%c0 to %c128 for %c128] : i32 -> %capture as !stream.resource<transient>{%size}
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[OPERAND]]
  util.optimization_barrier %result : !stream.resource<transient>
  util.return
}

// -----

// Tests that subviews of tied operands are propagated to consumers.

// CHECK-LABEL: @tiedOperandSubviews
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<external>,
// CHECK-SAME:  %[[SIZE:.+]]: index, %[[OFFSET0:.+]]: index, %[[OFFSET1:.+]]: index, %[[OFFSET2:.+]]: index, %[[LENGTH0:.+]]: index, %[[LENGTH1:.+]]: index, %[[LENGTH2:.+]]: index)
util.func public @tiedOperandSubviews(%operand: !stream.resource<external>, %size: index, %offset0: index, %offset1: index, %offset2: index, %length0: index, %length1: index, %length2: index) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[SUBVIEW_OFFSET:.+]] = arith.addi %[[OFFSET0]], %[[OFFSET1]]
  // CHECK: %[[SUBVIEW:.+]] = stream.resource.subview %[[OPERAND]][%[[SUBVIEW_OFFSET]]] {{.*}} -> !stream.resource<external>{%[[LENGTH1]]}
  // CHECK: stream.cmd.execute with(%[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<external>{%[[SIZE]]})
  // CHECK-NEXT: } => !stream.timepoint
  %result0, %result0_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<external>{%size}) -> (%operand as !stream.resource<external>{%length1}) {
    %subview0 = stream.resource.subview %capture[%offset0] : !stream.resource<external>{%size} -> !stream.resource<external>{%length0}
    %subview1 = stream.resource.subview %subview0[%offset1] : !stream.resource<external>{%length0} -> !stream.resource<external>{%length1}
    stream.yield %subview1 : !stream.resource<external>{%length1}
  } => !stream.timepoint
  // CHECK: stream.cmd.execute with(%[[SUBVIEW]] as %[[SUBVIEW_CAPTURE:.+]]: !stream.resource<external>{%[[LENGTH1]]})
  // CHECK: stream.cmd.copy %[[SUBVIEW_CAPTURE]][%[[OFFSET2]]], %[[SUBVIEW_CAPTURE]][%c0], %[[LENGTH2]] : !stream.resource<external>{%[[LENGTH1]]} -> !stream.resource<external>{%[[LENGTH1]]}
  %result1, %result1_timepoint = stream.async.execute with(%result0 as %capture: !stream.resource<external>{%length1}) -> (%result0 as !stream.resource<external>{%length1}) {
    %subview2 = stream.resource.subview %capture[%offset2] : !stream.resource<external>{%length1} -> !stream.resource<external>{%length2}
    %update = stream.async.update %subview2, %capture[%c0 to %length2] : !stream.resource<external>{%length2} -> %capture as !stream.resource<external>{%length1}
    stream.yield %update : !stream.resource<external>{%length1}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[SUBVIEW]]
  util.optimization_barrier %result1 : !stream.resource<external>
  util.return
}

// -----

// Tests that value aliases with subranges are propagated.

// CHECK-LABEL: @aliasPropagation
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<external>,
// CHECK-SAME:  %[[SIZE:.+]]: index, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index)
util.func public @aliasPropagation(%operand: !stream.resource<external>, %size: index, %offset: index, %length: index) {
  %c0 = arith.constant 0 : index
  // CHECK: stream.cmd.execute with(%[[OPERAND]] as %[[CAPTURE:.+]]: !stream.resource<external>{%[[SIZE]]})
  %result, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<external>{%size}) -> (%operand as !stream.resource<external>{%size}) {
    // CHECK-NOT: stream.resource.subview
    %subview = stream.resource.subview %capture[%offset] : !stream.resource<external>{%size} -> !stream.resource<external>{%length}
    // CHECK: stream.cmd.copy %[[CAPTURE]][%[[OFFSET]]], %[[CAPTURE]][%c0], %[[LENGTH]] : !stream.resource<external>{%[[SIZE]]} -> !stream.resource<external>{%[[SIZE]]}
    %update = stream.async.update %subview, %capture[%c0 to %length] : !stream.resource<external>{%length} -> %capture as !stream.resource<external>{%size}
    stream.yield %update : !stream.resource<external>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[OPERAND]]
  util.optimization_barrier %result : !stream.resource<external>
  util.return
}

// -----

// Tests results are allocated for external use.
// We expect them to be allocated with synchronous alloc ops.

// CHECK-LABEL: @producedResults
// CHECK-SAME: (%[[SIZE0:.+]]: index, %[[SIZE1:.+]]: index)
util.func public @producedResults(%size0: index, %size1: index) {
  %c254_i32 = arith.constant 254 : i32
  %c255_i32 = arith.constant 255 : i32
  //      CHECK: %[[PACK:.+]]:3 = stream.resource.pack slices({
  // CHECK-NEXT:   [0, 0] = %[[SIZE0]],
  // CHECK-NEXT:   [0, 0] = %[[SIZE1]]
  // CHECK-NEXT: }) : index
  // CHECK: %[[ALLOCA:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca uninitialized : !stream.resource<transient>{%[[PACK]]#0}
  // CHECK: %[[SUBALLOCA0:.+]] = stream.resource.subview %[[ALLOCA]][%[[PACK]]#1] : !stream.resource<transient>{%[[PACK]]#0} -> !stream.resource<transient>{%[[SIZE0]]}
  // CHECK: %[[SUBALLOCA1:.+]] = stream.resource.subview %[[ALLOCA]][%[[PACK]]#2] : !stream.resource<transient>{%[[PACK]]#0} -> !stream.resource<transient>{%[[SIZE1]]}
  // CHECK: %[[EXECUTE_TIMEPOINT:.+]] = stream.cmd.execute await(%[[ALLOCA_TIMEPOINT]])
  // CHECK-SAME: with(%[[SUBALLOCA0]] as %[[CAPTURE0:.+]]: !stream.resource<transient>{%[[SIZE0]]},
  // CHECK-SAME:      %[[SUBALLOCA1]] as %[[CAPTURE1:.+]]: !stream.resource<transient>{%[[SIZE1]]})
  %results:2, %result_timepoint = stream.async.execute with() -> (!stream.resource<transient>{%size0}, !stream.resource<transient>{%size1}) {
    // CHECK: stream.cmd.fill %c254_i32, %[[CAPTURE0]]
    %0 = stream.async.splat %c254_i32 : i32 -> !stream.resource<transient>{%size0}
    // CHECK: stream.cmd.fill %c255_i32, %[[CAPTURE1]]
    %1 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%size1}
    stream.yield %0, %1 : !stream.resource<transient>{%size0}, !stream.resource<transient>{%size1}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[EXECUTE_TIMEPOINT]]
  util.optimization_barrier %result_timepoint : !stream.timepoint
  // CHECK: util.optimization_barrier %[[SUBALLOCA0]]
  util.optimization_barrier %results#0 : !stream.resource<transient>
  // CHECK: util.optimization_barrier %[[SUBALLOCA1]]
  util.optimization_barrier %results#1 : !stream.resource<transient>
  util.return
}

// -----

// Tests local values that are produced and consumed exclusively within the
// execution region. We expect them to be placed into packed slices and
// allocated with the async stream-ordered alloca/dealloca ops.

util.global private @device : !hal.device

// CHECK-LABEL: @locals
// CHECK-SAME: (%[[SIZE0:.+]]: index, %[[SIZE1:.+]]: index, %[[AWAIT_TIMEPOINT:.+]]: !stream.timepoint)
util.func public @locals(%size0: index, %size1: index, %await_timepoint: !stream.timepoint) -> !stream.timepoint {
  %c254_i32 = arith.constant 254 : i32
  %c255_i32 = arith.constant 255 : i32
  //      CHECK: %[[SLICES:.+]]:3 = stream.resource.pack on(#hal.device.affinity<@device>) slices({
  // CHECK-NEXT:   [0, 0] = %[[SIZE0]],
  // CHECK-NEXT:   [1, 1] = %[[SIZE1]]
  // CHECK-NEXT: })
  // CHECK-NEXT: %[[ALLOCA:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca uninitialized on(#hal.device.affinity<@device>) await(%[[AWAIT_TIMEPOINT]]) => !stream.resource<transient>{%[[SLICES]]#0} => !stream.timepoint
  // CHECK-NEXT: %[[AWAIT_JOIN:.+]] = stream.timepoint.join max(%[[AWAIT_TIMEPOINT]], %[[ALLOCA_TIMEPOINT]])
  // CHECK: %[[EXEC_TIMEPOINT:.+]] = stream.cmd.execute on(#hal.device.affinity<@device>) await(%[[AWAIT_JOIN]])
  // CHECK-SAME: with(%[[ALLOCA]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[SLICES]]#0})
  %result_timepoint = stream.async.execute on(#hal.device.affinity<@device>) await(%await_timepoint) => with() {
    // CHECK: stream.cmd.fill %c254_i32, %[[CAPTURE]][%[[SLICES]]#1 for %[[SIZE0]]] : i32 -> !stream.resource<transient>{%[[SLICES]]#0}
    %0 = stream.async.splat %c254_i32 : i32 -> !stream.resource<transient>{%size0}
    // CHECK: stream.cmd.fill %c255_i32, %[[CAPTURE]][%[[SLICES]]#2 for %[[SIZE1]]] : i32 -> !stream.resource<transient>{%[[SLICES]]#0}
    %1 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%size1}
    stream.yield
  } => !stream.timepoint
  // CHECK: %[[DEALLOCA_TIMEPOINT:.+]] = stream.resource.dealloca on(#hal.device.affinity<@device>) await(%[[EXEC_TIMEPOINT]]) => %[[ALLOCA]] : !stream.resource<transient>{%[[SLICES]]#0} => !stream.timepoint
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[DEALLOCA_TIMEPOINT]], %[[EXEC_TIMEPOINT]]) => !stream.timepoint
  // CHECK: util.return %[[JOIN]]
  util.return %result_timepoint : !stream.timepoint
}

// -----

// Tests that concurrently executable regions don't introduce new allocations.
// They should effectively be no-ops with respect to allocation so this looks
// a lot like like the above tests.

// CHECK-LABEL: @concurrentRegions
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
util.func public @concurrentRegions(%operand: !stream.resource<transient>, %size: index) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c254_i32 = arith.constant 254 : i32
  %c255_i32 = arith.constant 255 : i32
  // CHECK: %[[ALLOCA:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca uninitialized : !stream.resource<transient>{%[[SIZE]]}
  // CHECK: stream.cmd.execute await(%[[ALLOCA_TIMEPOINT]])
  // CHECK-SAME: with(%[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOCA]] as %[[ALLOC_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
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
  // CHECK: util.optimization_barrier %[[ALLOCA]]
  util.optimization_barrier %results#1 : !stream.resource<transient>
  util.return
}

// -----

// CHECK-LABEL: @applyAsyncSplatOp
// CHECK-SAME: (%[[SIZE:.+]]: index)
util.func public @applyAsyncSplatOp(%size: index) {
  %c255_i32 = arith.constant 255 : i32
  // CHECK: %[[ALLOCA:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca uninitialized : !stream.resource<transient>{%[[SIZE]]}
  // CHECK: stream.cmd.execute await(%[[ALLOCA_TIMEPOINT]])
  // CHECK-SAME: with(%[[ALLOCA]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %result, %result_timepoint = stream.async.execute with() -> (!stream.resource<transient>{%size}) {
    // CHECK: stream.cmd.fill %c255_i32, %[[CAPTURE]][%c0 for %[[SIZE]]] : i32 -> !stream.resource<transient>{%[[SIZE]]}
    %0 = stream.async.splat %c255_i32 : i32 -> !stream.resource<transient>{%size}
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[ALLOCA]]
  util.optimization_barrier %result : !stream.resource<transient>
  util.return
}

// -----

// CHECK-LABEL: @applyAsyncCloneOp
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
util.func public @applyAsyncCloneOp(%operand: !stream.resource<transient>, %size: index) {
  // CHECK: %[[ALLOCA:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca uninitialized : !stream.resource<transient>{%[[SIZE]]}
  // CHECK: stream.cmd.execute await(%[[ALLOCA_TIMEPOINT]])
  // CHECK-SAME: with(%[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOCA]] as %[[ALLOC_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %result, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
    // CHECK: stream.cmd.copy %[[OPERAND_CAPTURE]][%c0], %[[ALLOC_CAPTURE]][%c0], %[[SIZE]]
    // CHECK-SAME: : !stream.resource<transient>{%[[SIZE]]} -> !stream.resource<transient>{%[[SIZE]]}
    %0 = stream.async.clone %capture : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size}
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[ALLOCA]]
  util.optimization_barrier %result : !stream.resource<transient>
  util.return
}

// -----

// TODO(benvanik): place the allocation instead.
// NOTE: this should be placing the allocation but is currently a copy.

// CHECK-LABEL: @applyAsyncSliceOp
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
util.func public @applyAsyncSliceOp(%operand: !stream.resource<transient>, %size: index) {
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %c144 = arith.constant 144 : index
  // CHECK: %[[ALLOCA:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca uninitialized : !stream.resource<transient>{%c128}
  // CHECK: stream.cmd.execute await(%[[ALLOCA_TIMEPOINT]])
  // CHECK-SAME: with(%[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOCA]] as %[[ALLOC_CAPTURE:.+]]: !stream.resource<transient>{%c128})
  %result, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%c128} {
    // CHECK: stream.cmd.copy %[[OPERAND_CAPTURE]][%c16], %[[ALLOC_CAPTURE]][%c0], %c128
    // CHECK-SAME: : !stream.resource<transient>{%[[SIZE]]} -> !stream.resource<transient>{%c128}
    %0 = stream.async.slice %capture[%c16 to %c144] : !stream.resource<transient>{%size} -> !stream.resource<transient>{%c128}
    stream.yield %0 : !stream.resource<transient>{%c128}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[ALLOCA]]
  util.optimization_barrier %result : !stream.resource<transient>
  util.return
}

// -----

// CHECK-LABEL: @applyAsyncFillOp
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
util.func public @applyAsyncFillOp(%operand: !stream.resource<transient>, %size: index) {
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
  util.return
}

// -----

// TODO(benvanik): place the allocation instead.
// NOTE: this should be placing the allocation but is currently a copy.

// CHECK-LABEL: @applyAsyncUpdateOp
// CHECK-SAME: (%[[UPDATE:.+]]: !stream.resource<external>,
// CHECK-SAME:  %[[OPERAND:.+]]: !stream.resource<transient>,
// CHECK-SAME:  %[[SIZE:.+]]: index)
util.func public @applyAsyncUpdateOp(%update: !stream.resource<external>, %operand: !stream.resource<transient>, %size: index) {
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
  util.return
}

// -----

// CHECK-LABEL: @applyAsyncCopyOp
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<external>,
// CHECK-SAME:  %[[TARGET:.+]]: !stream.resource<transient>,
// CHECK-SAME:  %[[SIZE:.+]]: index)
util.func public @applyAsyncCopyOp(%source: !stream.resource<external>, %target: !stream.resource<transient>, %size: index) {
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
  util.return
}

// -----

// Tests that fully in-place execution regions with nesting don't allocate any
// transients. Both copies are concurrently in-place to the same provided
// target buffer.

// CHECK-LABEL: @applyConcurrentAsyncCopyOp
// CHECK-SAME: (%[[SOURCE:.+]]: !stream.resource<external>,
// CHECK-SAME:  %[[TARGET:.+]]: !stream.resource<transient>,
// CHECK-SAME:  %[[SIZE:.+]]: index)
util.func public @applyConcurrentAsyncCopyOp(%source: !stream.resource<external>, %target: !stream.resource<transient>, %size: index) {
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
  util.return
}

// -----

// TODO(#11249): add a test for in-place collectives (send == recv).

// CHECK-LABEL: @applyAsyncCollectiveOpOutOfPlace
// CHECK-SAME: (%[[CHANNEL:.+]]: !stream.channel,
// CHECK-SAME:  %[[SEND:.+]]: !stream.resource<external>, %[[SEND_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[RECV:.+]]: !stream.resource<transient>, %[[RECV_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[COUNT:[a-z0-9]+]]: index)
util.func public @applyAsyncCollectiveOpOutOfPlace(%channel: !stream.channel, %send: !stream.resource<external>, %send_size: index, %recv: !stream.resource<transient>, %recv_size: index, %count: index) {
  %c0 = arith.constant 0 : index
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
  util.return
}

// -----

// TODO(benvanik): test affinity changes that would introduce invalidate/fill.

// CHECK-LABEL: @applyAsyncTransferOp
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
util.func public @applyAsyncTransferOp(%operand: !stream.resource<transient>, %size: index) {
  // CHECK: %[[ALLOCA:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca uninitialized : !stream.resource<transient>{%[[SIZE]]}
  // CHECK: stream.cmd.execute await(%[[ALLOCA_TIMEPOINT]])
  // CHECK-SAME: with(%[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOCA]] as %[[ALLOCA_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %result, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
    // CHECK: stream.cmd.copy %[[OPERAND_CAPTURE]][%c0], %[[ALLOCA_CAPTURE]][%c0], %[[SIZE]]
    // CHECK-SAME: : !stream.resource<transient>{%[[SIZE]]} -> !stream.resource<transient>{%[[SIZE]]}
    %0 = stream.async.transfer %capture : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size}
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[ALLOCA]]
  util.optimization_barrier %result : !stream.resource<transient>
  util.return
}

// -----

// CHECK-LABEL: @applyAsyncTransferMultiScopeOp
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
util.func public @applyAsyncTransferMultiScopeOp(%operand: !stream.resource<transient>, %size: index) {
  // CHECK: %[[ALLOCA:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@execution_device>, #hal.device.affinity<@result_device>]>) : !stream.resource<transient>{%[[SIZE]]}
  // CHECK: stream.cmd.execute on(#hal.device.affinity<@execution_device>) await(%[[ALLOCA_TIMEPOINT]])
  // CHECK-SAME: with(%[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOCA]] as %[[ALLOCA_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %result, %result_timepoint = stream.async.execute on(#hal.device.affinity<@execution_device>) with(%operand as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
    // CHECK: stream.cmd.copy %[[OPERAND_CAPTURE]][%c0], %[[ALLOCA_CAPTURE]][%c0], %[[SIZE]]
    // CHECK-SAME: : !stream.resource<transient>{%[[SIZE]]} -> !stream.resource<transient>{%[[SIZE]]}
    // CHECK: stream.cmd.flush to(#hal.device.affinity<@result_device>) %[[ALLOCA_CAPTURE]][%c0 for %[[SIZE]]]
    // CHECK-SAME: : !stream.resource<transient>{%[[SIZE]]}
    %0 = stream.async.transfer %capture : !stream.resource<transient>{%size} -> to(#hal.device.affinity<@result_device>) !stream.resource<transient>{%size}
    stream.yield %0 : !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[ALLOCA]]
  util.optimization_barrier %result : !stream.resource<transient>
  util.return
}

// -----

// CHECK-LABEL: @applyAsyncConcurrentTransferMultiScopeOp
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
util.func public @applyAsyncConcurrentTransferMultiScopeOp(%operand: !stream.resource<transient>, %size: index) {
  // CHECK-DAG: %[[ALLOCA_A:.+]], %[[ALLOCA_A_TIMEPOINT:.+]] = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@execution_device>, #hal.device.affinity<@result_device_a>]>) : !stream.resource<transient>{%[[SIZE]]}
  // CHECK-DAG: %[[ALLOCA_B:.+]], %[[ALLOCA_B_TIMEPOINT:.+]] = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@execution_device>, #hal.device.affinity<@result_device_b>]>) : !stream.resource<transient>{%[[SIZE]]}
  // CHECK-DAG: %[[ALLOCA_TIMEPOINTS:.+]] = stream.timepoint.join max(%[[ALLOCA_A_TIMEPOINT]], %[[ALLOCA_B_TIMEPOINT]])
  // CHECK: stream.cmd.execute on(#hal.device.affinity<@execution_device>) await(%[[ALLOCA_TIMEPOINTS]])
  // CHECK-SAME: with(%[[OPERAND]] as %[[OPERAND_CAPTURE:[a-z0-9_]+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOCA_A]] as %[[ALLOCA_A_CAPTURE:[a-z0-9_]+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOCA_B]] as %[[ALLOCA_B_CAPTURE:[a-z0-9_]+]]: !stream.resource<transient>{%[[SIZE]]})
  %results:2, %result_timepoint = stream.async.execute on(#hal.device.affinity<@execution_device>) with(%operand as %capture: !stream.resource<transient>{%size}) -> (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) {
    // CHECK: stream.cmd.concurrent
    %concurrent:2 = stream.async.concurrent with(%capture as %concurrent_capture: !stream.resource<transient>{%size}) -> (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) {
      // CHECK: stream.cmd.copy %[[OPERAND_CAPTURE]][%c0], %[[ALLOCA_A_CAPTURE]][%c0], %[[SIZE]]
      // CHECK-SAME: : !stream.resource<transient>{%[[SIZE]]} -> !stream.resource<transient>{%[[SIZE]]}
      // CHECK: stream.cmd.flush to(#hal.device.affinity<@result_device_a>) %[[ALLOCA_A_CAPTURE]][%c0 for %[[SIZE]]]
      // CHECK-SAME: : !stream.resource<transient>{%[[SIZE]]}
      %transfer_a = stream.async.transfer %concurrent_capture : !stream.resource<transient>{%size} -> to(#hal.device.affinity<@result_device_a>) !stream.resource<transient>{%size}
      // CHECK: stream.cmd.copy %[[OPERAND_CAPTURE]][%c0], %[[ALLOCA_B_CAPTURE]][%c0], %[[SIZE]]
      // CHECK-SAME: : !stream.resource<transient>{%[[SIZE]]} -> !stream.resource<transient>{%[[SIZE]]}
      // CHECK: stream.cmd.flush to(#hal.device.affinity<@result_device_b>) %[[ALLOCA_B_CAPTURE]][%c0 for %[[SIZE]]]
      // CHECK-SAME: : !stream.resource<transient>{%[[SIZE]]}
      %transfer_b = stream.async.transfer %concurrent_capture : !stream.resource<transient>{%size} -> to(#hal.device.affinity<@result_device_b>) !stream.resource<transient>{%size}
      stream.yield %transfer_a, %transfer_b : !stream.resource<transient>{%size}, !stream.resource<transient>{%size}
    }
    stream.yield %concurrent#0, %concurrent#1 : !stream.resource<transient>{%size}, !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[ALLOCA_A]]
  util.optimization_barrier %results#0 : !stream.resource<transient>
  // CHECK: util.optimization_barrier %[[ALLOCA_B]]
  util.optimization_barrier %results#1 : !stream.resource<transient>
  util.return
}

// -----

// CHECK-LABEL: @applyAsyncDispatchOp
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[OFFSET:.+]]: index, %[[END:.+]]: index, %[[LENGTH:.+]]: index)
util.func public @applyAsyncDispatchOp(%operand: !stream.resource<transient>, %size: index, %offset: index, %end: index, %length: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[ALLOCA:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca uninitialized : !stream.resource<transient>{%[[SIZE]]}
  // CHECK: %[[TIMEPOINT:.+]] = stream.cmd.execute await(%[[ALLOCA_TIMEPOINT]])
  // CHECK-SAME: with(%[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOCA]] as %[[ALLOCA_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %results:2, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> (%operand as !stream.resource<transient>{%size}, !stream.resource<transient>{%size}) {
    // CHECK-NEXT: stream.cmd.dispatch @executable::@dispatch[%c1, %c1, %c1](%c4 : index) {
    // CHECK-NEXT:   rw %[[OPERAND_CAPTURE]][%[[OFFSET]] for %[[LENGTH]]] : !stream.resource<transient>{%[[SIZE]]},
    // CHECK-NEXT:   wo %[[ALLOCA_CAPTURE]][%c0{{[_0-9]*}} for %[[SIZE]]] : !stream.resource<transient>{%[[SIZE]]}
    // CHECK-NEXT: }
    %0:2 = stream.async.dispatch @executable::@dispatch[%c1, %c1, %c1](%capture[%offset to %end for %length], %c4) : (!stream.resource<transient>{%size}, index) -> (%capture{%size}, !stream.resource<transient>{%size})
    stream.yield %0#0, %0#1 : !stream.resource<transient>{%size}, !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[TIMEPOINT]]
  util.optimization_barrier %result_timepoint : !stream.timepoint
  // CHECK: util.optimization_barrier %[[OPERAND]]
  util.optimization_barrier %results#0 : !stream.resource<transient>
  // CHECK: util.optimization_barrier %[[ALLOCA]]
  util.optimization_barrier %results#1 : !stream.resource<transient>
  util.return
}

// -----

// Tests that unused dispatch results nested in concurrent regions are still
// allocated memory.

// CHECK-LABEL: @applyAsyncDispatchUnusedOp
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[OFFSET:.+]]: index, %[[END:.+]]: index, %[[LENGTH:.+]]: index)
util.func public @applyAsyncDispatchUnusedOp(%operand: !stream.resource<transient>, %size: index, %offset: index, %end: index, %length: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[PACK:.+]]:2 = stream.resource.pack
  // CHECK: %[[ALLOCA:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca uninitialized : !stream.resource<transient>{%[[PACK]]#0}
  // CHECK: %[[TIMEPOINT:.+]] = stream.cmd.execute
  // CHECK-SAME: await(%[[ALLOCA_TIMEPOINT]])
  // CHECK-SAME: with(%[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOCA]] as %[[ALLOCA_CAPTURE:.+]]: !stream.resource<transient>{%[[PACK]]#0})
  %result, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> (%operand as !stream.resource<transient>{%size}) {
    // CHECK: stream.cmd.concurrent
    %concurrent = stream.async.concurrent with(%capture as %concurrent_capture: !stream.resource<transient>{%size}) -> (%capture as !stream.resource<transient>{%size}) {
      // CHECK-NEXT: stream.cmd.dispatch @executable::@dispatch[%c1, %c1, %c1](%c4 : index) {
      // CHECK-NEXT:   rw %[[OPERAND_CAPTURE]][%[[OFFSET]] for %[[LENGTH]]] : !stream.resource<transient>{%[[SIZE]]},
      // CHECK-NEXT:   wo %[[ALLOCA_CAPTURE]][%[[PACK]]#1 for %[[SIZE]]] : !stream.resource<transient>{%[[PACK]]#0}
      // CHECK-NEXT: }
      %0:2 = stream.async.dispatch @executable::@dispatch[%c1, %c1, %c1](%concurrent_capture[%offset to %end for %length], %c4) : (!stream.resource<transient>{%size}, index) -> (%concurrent_capture{%size}, !stream.resource<transient>{%size})
      // NOTE: %0#1 is unused.
      stream.yield %0#0 : !stream.resource<transient>{%size}
    }
    stream.yield %concurrent : !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: %[[DEALLOCA:.+]] = stream.resource.dealloca await(%[[TIMEPOINT]]) => %[[ALLOCA]]
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[DEALLOCA]], %[[TIMEPOINT]])
  // CHECK: util.optimization_barrier %[[JOIN]]
  util.optimization_barrier %result_timepoint : !stream.timepoint
  // CHECK: util.optimization_barrier %[[OPERAND]]
  util.optimization_barrier %result : !stream.resource<transient>
  util.return
}

// -----

// Tests that unused dispatch results that are tied do not get their own local
// allocation.

// CHECK-LABEL: @applyAsyncDispatchUnusedTiedOperand
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[OFFSET:.+]]: index, %[[END:.+]]: index, %[[LENGTH:.+]]: index)
util.func public @applyAsyncDispatchUnusedTiedOperand(%operand: !stream.resource<transient>, %size: index, %offset: index, %end: index, %length: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[PACK:.+]]:2 = stream.resource.pack
  // CHECK: %[[ALLOCA:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca uninitialized : !stream.resource<transient>{%[[PACK]]#0}
  // CHECK: %[[TIMEPOINT:.+]] = stream.cmd.execute
  // CHECK-SAME: await(%[[ALLOCA_TIMEPOINT]])
  // CHECK-SAME: with(%[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOCA]] as %[[ALLOCA_CAPTURE:.+]]: !stream.resource<transient>{%[[PACK]]#0})
  %result, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> (%operand as !stream.resource<transient>{%size}) {
    // CHECK: stream.cmd.concurrent
    %alloca = stream.async.alloca : !stream.resource<transient>{%size}
    %concurrent = stream.async.concurrent with(%capture as %concurrent_capture: !stream.resource<transient>{%size},
        %alloca as %unused_capture: !stream.resource<transient>{%size}) -> (%capture as !stream.resource<transient>{%size}) {
      // CHECK-NEXT: stream.cmd.dispatch @executable::@dispatch[%c1, %c1, %c1](%c4 : index) {
      // CHECK-NEXT:   rw %[[OPERAND_CAPTURE]][%[[OFFSET]] for %[[LENGTH]]] : !stream.resource<transient>{%[[SIZE]]},
      // CHECK-NEXT:   rw %[[ALLOCA_CAPTURE]][%[[PACK]]#1 for %[[SIZE]]] : !stream.resource<transient>{%[[PACK]]#0}
      // CHECK-NEXT: }
      %0:2 = stream.async.dispatch @executable::@dispatch[%c1, %c1, %c1](%concurrent_capture[%offset to %end for %length], %unused_capture[%c0 to %size for %size], %c4)
        : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}, index) -> (%concurrent_capture{%size}, %unused_capture{%size})
      // NOTE: %0#1 is unused.
      stream.yield %0#0 : !stream.resource<transient>{%size}
    }
    stream.yield %concurrent : !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: %[[DEALLOCA:.+]] = stream.resource.dealloca await(%[[TIMEPOINT]]) => %[[ALLOCA]]
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[DEALLOCA]], %[[TIMEPOINT]])
  // CHECK: util.optimization_barrier %[[JOIN]]
  util.optimization_barrier %result_timepoint : !stream.timepoint
  // CHECK: util.optimization_barrier %[[OPERAND]]
  util.optimization_barrier %result : !stream.resource<transient>
  util.return
}

// -----

// CHECK: stream.cmd.func private @asyncExtern(%arg0[%arg1 for %arg2]: !stream.resource<transient>, %arg3: index, %arg4[%arg5 for %arg6]: !stream.resource<transient>)
stream.async.func private @asyncExtern(%arg0: !stream.resource<transient>, %arg1: index) -> (%arg0, !stream.resource<transient>)

// CHECK-LABEL: @applyAsyncCallOp
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[OFFSET:.+]]: index, %[[END:.+]]: index, %[[LENGTH:.+]]: index)
util.func public @applyAsyncCallOp(%operand: !stream.resource<transient>, %size: index, %offset: index, %end: index, %length: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[ALLOCA:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca uninitialized : !stream.resource<transient>{%[[SIZE]]}
  // CHECK: %[[TIMEPOINT:.+]] = stream.cmd.execute await(%[[ALLOCA_TIMEPOINT]])
  // CHECK-SAME: with(%[[OPERAND]] as %[[OPERAND_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]},
  // CHECK-SAME:      %[[ALLOCA]] as %[[ALLOCA_CAPTURE:.+]]: !stream.resource<transient>{%[[SIZE]]})
  %results:2, %result_timepoint = stream.async.execute with(%operand as %capture: !stream.resource<transient>{%size}) -> (%operand as !stream.resource<transient>{%size}, !stream.resource<transient>{%size}) {
    // CHECK-NEXT: stream.cmd.call @asyncExtern(rw %[[OPERAND_CAPTURE]][%[[OFFSET]] for %[[LENGTH]]], %c4, wo %[[ALLOCA_CAPTURE]][%c0{{[_0-9]*}} for %[[SIZE]]]) :
    // CHECK-SAME:     (!stream.resource<transient>{%[[SIZE]]}, index, !stream.resource<transient>{%[[SIZE]]}) -> ()
    %0:2 = stream.async.call @asyncExtern(%capture[%offset to %end for %length], %c4) : (!stream.resource<transient>{%size}, index) -> (%capture{%size}, !stream.resource<transient>{%size})
    stream.yield %0#0, %0#1 : !stream.resource<transient>{%size}, !stream.resource<transient>{%size}
  } => !stream.timepoint
  // CHECK: util.optimization_barrier %[[TIMEPOINT]]
  util.optimization_barrier %result_timepoint : !stream.timepoint
  // CHECK: util.optimization_barrier %[[OPERAND]]
  util.optimization_barrier %results#0 : !stream.resource<transient>
  // CHECK: util.optimization_barrier %[[ALLOCA]]
  util.optimization_barrier %results#1 : !stream.resource<transient>
  util.return
}

// -----

// Tests that stream.async.load/store are converted to their explicit forms.

// CHECK-LABEL: @asyncLoadStore
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<staging>,
// CHECK-SAME:  %[[SIZE:.+]]: index)
util.func public @asyncLoadStore(%operand: !stream.resource<staging>, %size: index) -> f32 {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 5.4 : f32
  // CHECK: stream.resource.store %cst, %[[OPERAND]][%c0] : f32 -> !stream.resource<staging>{%[[SIZE]]}
  %0 = stream.async.store %cst, %operand[%c0] : f32 -> %operand as !stream.resource<staging>{%size}
  // CHECK: %[[RESULT:.+]] = stream.resource.load %[[OPERAND]][%c0] : !stream.resource<staging>{%[[SIZE]]} -> f32
  %1 = stream.async.load %0[%c0] : !stream.resource<staging>{%size} -> f32
  // CHECK: util.return %[[RESULT]]
  util.return %1 : f32
}

// -----

// Tests that we recurse into scf operations.

// CHECK-LABEL: @scfFor
// CHECK-SAME: (%[[OPERAND:.+]]: !stream.resource<staging>,
// CHECK-SAME:  %[[SIZE:.+]]: index)
util.func public @scfFor(%operand: !stream.resource<staging>, %size: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32

  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[C2:.+]] = arith.constant 2 : index
  // CHECK: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: %[[ZERO:.+]] = arith.constant 0.0
  // CHECK: %[[FOR:.+]] = scf.for %[[ARG2:.+]] = %[[C0]] to %[[C4]] step %[[C2]] iter_args(%[[ARG3:.+]] = %[[ZERO]])
  %sum = scf.for %i = %c0 to %c4 step %c2 iter_args(%arg0 = %zero) -> f32 {
  // CHECK: %[[RESULT:.+]] = stream.resource.load %[[OPERAND]][%[[ARG2]]] : !stream.resource<staging>{%[[SIZE]]} -> f32
    %0 = stream.async.load %operand[%i] : !stream.resource<staging>{%size} -> f32
  // CHECK: stream.resource.store %cst, %[[OPERAND]][%[[ARG2]]] : f32 -> !stream.resource<staging>{%[[SIZE]]}
    %1 = stream.async.store %zero, %operand[%i] : f32 -> %operand as !stream.resource<staging>{%size}
    %2 = arith.addf %0, %arg0 : f32
    scf.yield %2 : f32
  }

  // CHECK: util.return %[[FOR]]
  util.return %sum : f32
}

// -----

// Tests round-tripping a buffer through multiple devices.
// We expect the allocations to be made optimal for both devices.

// CHECK-LABEL: @multiAffinityTrip
// CHECK-SAME: (%[[INPUT_TIMEPOINT:.+]]: !stream.timepoint
util.func public @multiAffinityTrip(%input_timepoint: !stream.timepoint, %input_handle: i64) -> (!stream.timepoint, i64) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  // CHECK: %[[INPUT:.+]] = stream.tensor.import
  %input = stream.tensor.import on(#hal.device.promise<@device_a>) %input_handle : i64 -> tensor<4xf32> in !stream.resource<external>{%c16}
  // CHECK: %[[RESULT_A:.+]], %[[RESULT_A_TIMEPOINT:.+]] = stream.resource.alloca
  // CHECK-SAME: on(#hal.device.optimal<[#hal.device.promise<@device_a>, #hal.device.promise<@device_b>]>)
  // CHECK-SAME: await(%[[INPUT_TIMEPOINT]])
  // CHECK: %[[EXECUTE_A_READY:.+]] = stream.timepoint.join max(%[[INPUT_TIMEPOINT]], %[[RESULT_A_TIMEPOINT]])
  // CHECK: %[[TIMEPOINT_A:.+]] = stream.cmd.execute on(#hal.device.promise<@device_a>) await(%[[EXECUTE_A_READY]]) => with(%[[INPUT]]{{.+}}, %[[RESULT_A]]
  %result_a, %timepoint_a = stream.async.execute on(#hal.device.promise<@device_a>) await(%input_timepoint) => with(%input as %capture: !stream.resource<external>{%c16}) -> !stream.resource<transient>{%c16} {
    %t = stream.async.dispatch @executable::@dispatch_a(%capture[%c0 to %c16 for %c16]) : (!stream.resource<external>{%c16}) -> !stream.resource<transient>{%c16}
    stream.yield %t : !stream.resource<transient>{%c16}
  } => !stream.timepoint
  // CHECK: %[[RESULT_B:.+]], %[[RESULT_B_TIMEPOINT:.+]] = stream.resource.alloca
  // CHECK-SAME: on(#hal.device.optimal<[#hal.device.promise<@device_a>, #hal.device.promise<@device_b>]>)
  // CHECK-SAME: await(%[[TIMEPOINT_A]])
  // CHECK: %[[EXECUTE_B_READY:.+]] = stream.timepoint.join max(%[[TIMEPOINT_A]], %[[RESULT_B_TIMEPOINT]])
  // CHECK: %[[TIMEPOINT_B:.+]] = stream.cmd.execute on(#hal.device.promise<@device_b>) await(%[[EXECUTE_B_READY]]) => with(%[[RESULT_A]]{{.+}}, %[[RESULT_B]]
  %result_b, %timepoint_b = stream.async.execute on(#hal.device.promise<@device_b>) await(%timepoint_a) => with(%result_a as %capture: !stream.resource<transient>{%c16}) -> !stream.resource<external>{%c16} {
    %t = stream.async.dispatch @executable::@dispatch_b(%capture[%c0 to %c16 for %c16]) : (!stream.resource<transient>{%c16}) -> !stream.resource<external>{%c16}
    stream.yield %t : !stream.resource<external>{%c16}
  } => !stream.timepoint
  // CHECK: stream.tensor.export {{.+}} %[[RESULT_B]]
  %output_handle = stream.tensor.export on(#hal.device.promise<@device_a>) %result_b : tensor<4xf32> in !stream.resource<external>{%c16} -> i64
  // CHECK: util.return %[[TIMEPOINT_B]]
  util.return %timepoint_b, %output_handle : !stream.timepoint, i64
}
