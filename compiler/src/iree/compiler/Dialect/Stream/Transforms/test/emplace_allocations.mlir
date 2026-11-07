// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-stream-emplace-allocations,util.func(cse))' %s | FileCheck %s
// RUN: iree-opt --mlir-disable-threading --split-input-file --pass-pipeline='builtin.module(iree-stream-emplace-allocations,util.func(cse))' %s | FileCheck %s

// Tests that a dispatch result is placed into the target of an update.

// CHECK-LABEL: @emplaceDispatch
util.func public @emplaceDispatch(
    // CHECK-SAME: %[[INPUT:arg[0-9]+]]: !stream.resource<*>, %[[INPUT_SIZE:arg[0-9]+]]: index,
    %input: !stream.resource<*>, %input_size: index,
    // CHECK-SAME: %[[UPDATE_OFFSET:arg[0-9]+]]: index, %[[UPDATE_SIZE:arg[0-9]+]]: index,
    %update_offset: index, %update_size: index,
    // CHECK-SAME: %[[TARGET:arg[0-9]+]]: !stream.resource<*>, %[[TARGET_SIZE:arg[0-9]+]]: index
    %target: !stream.resource<*>, %target_size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[UNIFORM0:.+]] = arith.constant 123
  %uniform0 = arith.constant 123 : i32
  // CHECK-DAG: %[[UNIFORM1:.+]] = arith.constant 456
  %uniform1 = arith.constant 456 : i32
  // CHECK: %[[UPDATE_END:.+]] = arith.addi %[[UPDATE_OFFSET]], %[[UPDATE_SIZE]]
  // CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex::@dispatch[%c0](%[[UNIFORM0]], %[[INPUT]][%c0 to %[[INPUT_SIZE]] for %[[INPUT_SIZE]]], %[[UNIFORM1]], %[[TARGET]][%[[UPDATE_OFFSET]] to %[[UPDATE_END]] for %[[UPDATE_SIZE]]]) :
  // CHECK-SAME: (i32, !stream.resource<*>{%[[INPUT_SIZE]]}, i32, !stream.resource<*>{%[[TARGET_SIZE]]}) -> %[[TARGET]]{%[[TARGET_SIZE]]}
  %update = stream.async.dispatch @ex::@dispatch[%c0](%uniform0, %input[%c0 to %input_size for %input_size], %uniform1) : (i32, !stream.resource<*>{%input_size}, i32) -> !stream.resource<*>{%update_size}
  // NOTE: this gets hoisted above the dispatch.
  %update_end = arith.addi %update_offset, %update_size : index
  // CHECK-NOT: stream.async.update
  %result = stream.async.update %update, %target[%update_offset to %update_end] : !stream.resource<*>{%update_size} -> %target as !stream.resource<*>{%target_size}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// -----

// Tests that a dispatch with a tied result does not get placed. We could
// possibly untie the operand and place the result into the target resource but
// if the dispatch requires in-place operation that may not be safe.

// CHECK-LABEL: @dontEmplaceTiedDispatch
util.func public @dontEmplaceTiedDispatch(
    %tied_input: !stream.resource<*>, %tied_input_size: index,
    %update_offset: index, %update_size: index,
    %target: !stream.resource<*>, %target_size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %update_end = arith.addi %update_offset, %update_size : index
  // CHECK: %[[TIED_RESULT:.+]] = stream.async.dispatch @ex::@dispatch
  %update = stream.async.dispatch @ex::@dispatch(%tied_input[%c0 to %tied_input_size for %tied_input_size]) : (!stream.resource<*>{%tied_input_size}) -> %tied_input{%tied_input_size}
  // CHECK: %[[RESULT:.+]] = stream.async.update %[[TIED_RESULT]]
  %result = stream.async.update %update, %target[%update_offset to %update_end] : !stream.resource<*>{%tied_input_size} -> %target as !stream.resource<*>{%target_size}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// -----

// Tests that sequences of updates get reordered and placed.
// This pattern originates from concats in higher level representations and we
// test that explicitly as it's 95% of what this pass is designed to optimize.

// CHECK-LABEL: @emplaceDispatchSequence
util.func public @emplaceDispatchSequence(
    // CHECK-SAME: %[[INPUT:arg[0-9]+]]: !stream.resource<*>, %[[INPUT_SIZE:arg[0-9]+]]: index,
    %input: !stream.resource<*>, %input_size: index,
    // CHECK-SAME: %[[UPDATE_SIZE:arg[0-9]+]]: index, %[[TARGET_SIZE:arg[0-9]+]]: index
    %update_size: index, %target_size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c49152 = arith.constant 49152 : index
  %c98304 = arith.constant 98304 : index
  %c147456 = arith.constant 147456 : index
  %c196608 = arith.constant 196608 : index
  // CHECK: %[[TARGET:.+]] = stream.async.alloca : !stream.resource<*>{%[[TARGET_SIZE]]}
  // CHECK: %[[TARGET0:.+]] = stream.async.dispatch @ex::@dispatch0({{.+}}, %[[TARGET]][%c0 to %c49152 for %[[UPDATE_SIZE]]]) : ({{.+}}) -> %[[TARGET]]{%[[TARGET_SIZE]]}
  %update0 = stream.async.dispatch @ex::@dispatch0(%input[%c0 to %input_size for %input_size]) : (!stream.resource<*>{%input_size}) -> !stream.resource<*>{%update_size}
  // CHECK: %[[TARGET1:.+]] = stream.async.dispatch @ex::@dispatch1({{.+}}, %[[TARGET0]][%c49152 to %c98304 for %[[UPDATE_SIZE]]]) : ({{.+}}) -> %[[TARGET0]]{%[[TARGET_SIZE]]}
  %update1 = stream.async.dispatch @ex::@dispatch1(%input[%c0 to %input_size for %input_size]) : (!stream.resource<*>{%input_size}) -> !stream.resource<*>{%update_size}
  // CHECK: %[[TARGET2:.+]] = stream.async.dispatch @ex::@dispatch2({{.+}}, %[[TARGET1]][%c98304 to %c147456 for %[[UPDATE_SIZE]]]) : ({{.+}}) -> %[[TARGET1]]{%[[TARGET_SIZE]]}
  %update2 = stream.async.dispatch @ex::@dispatch2(%input[%c0 to %input_size for %input_size]) : (!stream.resource<*>{%input_size}) -> !stream.resource<*>{%update_size}
  // CHECK: %[[TARGET3:.+]] = stream.async.dispatch @ex::@dispatch3({{.+}}, %[[TARGET2]][%c147456 to %c196608 for %[[UPDATE_SIZE]]]) : ({{.+}}) -> %[[TARGET2]]{%[[TARGET_SIZE]]}
  %update3 = stream.async.dispatch @ex::@dispatch3(%input[%c0 to %input_size for %input_size]) : (!stream.resource<*>{%input_size}) -> !stream.resource<*>{%update_size}
  // CHECK-NOT: stream.async.alloca
  %target = stream.async.alloca : !stream.resource<*>{%target_size}
  // CHECK-NOT: stream.async.update
  %target0 = stream.async.update %update0, %target[%c0 to %c49152] : !stream.resource<*>{%update_size} -> %target as !stream.resource<*>{%target_size}
  // CHECK-NOT: stream.async.update
  %target1 = stream.async.update %update1, %target0[%c49152 to %c98304] : !stream.resource<*>{%update_size} -> %target0 as !stream.resource<*>{%target_size}
  // CHECK-NOT: stream.async.update
  %target2 = stream.async.update %update2, %target1[%c98304 to %c147456] : !stream.resource<*>{%update_size} -> %target1 as !stream.resource<*>{%target_size}
  // CHECK-NOT: stream.async.update
  %target3 = stream.async.update %update3, %target2[%c147456 to %c196608] : !stream.resource<*>{%update_size} -> %target2 as !stream.resource<*>{%target_size}
  // CHECK: util.return %[[TARGET3]]
  util.return %target3 : !stream.resource<*>
}

// -----

// Tests that emplacement propagation through tied results is stopped by
// hazards. Here the %temp0 in-place operation on %update0 stops the placement
// of %update1 into %target0.

// CHECK-LABEL: @dontEmplaceDispatchWithTiedResults
func.func @dontEmplaceDispatchWithTiedResults(
    // CHECK-SAME: %[[INPUT:arg[0-9]+]]: !stream.resource<*>, %[[INPUT_SIZE:arg[0-9]+]]: index,
    %input: !stream.resource<*>, %input_size: index,
    // CHECK-SAME: %[[UPDATE_SIZE:arg[0-9]+]]: index, %[[TARGET_SIZE:arg[0-9]+]]: index
    %update_size: index, %target_size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c49152 = arith.constant 49152 : index
  %c98304 = arith.constant 98304 : index
  // CHECK: stream.async.dispatch @ex::@dispatch0
  %update0 = stream.async.dispatch @ex::@dispatch0(%input[%c0 to %input_size for %input_size]) : (!stream.resource<*>{%input_size}) -> !stream.resource<*>{%update_size}
  // CHECK: stream.async.dispatch @ex::@dispatch1
  %temp0 = stream.async.dispatch @ex::@dispatch1(%update0[%c0 to %update_size for %update_size]) : (!stream.resource<*>{%update_size}) -> %update0{%update_size}
  // CHECK: stream.async.dispatch @ex::@dispatch2
  %update1 = stream.async.dispatch @ex::@dispatch2(%temp0[%c0 to %update_size for %update_size], %update0[%c0 to %update_size for %update_size]) : (!stream.resource<*>{%update_size}, !stream.resource<*>{%update_size}) -> %temp0{%update_size}
  // CHECK: stream.async.alloca
  %target = stream.async.alloca : !stream.resource<*>{%target_size}
  // CHECK: stream.async.update
  %target0 = stream.async.update %update0, %target[%c0 to %c49152] : !stream.resource<*>{%update_size} -> %target as !stream.resource<*>{%target_size}
  // CHECK: stream.async.update
  %target1 = stream.async.update %update1, %target0[%c49152 to %c98304] : !stream.resource<*>{%update_size} -> %target0 as !stream.resource<*>{%target_size}
  return %target1 : !stream.resource<*>
}

// -----

// Tests a concat-like sequence that has some inter-dependencies - these
// dependencies shouldn't stop us from emplacing.

// CHECK-LABEL: @emplaceMultiResultDispatchSequence
util.func public @emplaceMultiResultDispatchSequence(
    // CHECK-SAME: %[[INPUT:arg[0-9]+]]: !stream.resource<*>, %[[INPUT_SIZE:arg[0-9]+]]: index,
    %input: !stream.resource<*>, %input_size: index,
    // CHECK-SAME: %[[UPDATE_SIZE:arg[0-9]+]]: index, %[[TARGET_SIZE:arg[0-9]+]]: index
    %update_size: index, %target_size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c49152 = arith.constant 49152 : index
  %c98304 = arith.constant 98304 : index
  %c147456 = arith.constant 147456 : index
  %c196608 = arith.constant 196608 : index
  // CHECK: %[[TARGET:.+]] = stream.async.alloca : !stream.resource<*>{%[[TARGET_SIZE]]}
  // CHECK: %[[TARGET0:.+]] = stream.async.dispatch @ex::@dispatch0({{.+}}, %[[TARGET]][%c0 to %c49152 for %[[UPDATE_SIZE]]]) : ({{.+}}) -> %[[TARGET]]{%[[TARGET_SIZE]]}
  %update0 = stream.async.dispatch @ex::@dispatch0(%input[%c0 to %input_size for %input_size]) : (!stream.resource<*>{%input_size}) -> !stream.resource<*>{%update_size}
  // The second result gets allocated even though it's not emplaced due to how the pass works.
  // CHECK: %[[WRITE1_ALLOCA:.+]] = stream.async.alloca : !stream.resource<*>{%[[INPUT_SIZE]]}
  // CHECK: %[[TARGET1_TEMP:.+]]:2 = stream.async.dispatch @ex::@dispatch1({{.+}}, %[[TARGET0]][%c49152 to %c98304 for %[[UPDATE_SIZE]]], %[[WRITE1_ALLOCA]][%c0 to %[[INPUT_SIZE]] for %[[INPUT_SIZE]]]) : ({{.+}}) -> (%[[TARGET0]]{%[[TARGET_SIZE]]}, %[[WRITE1_ALLOCA]]{%[[INPUT_SIZE]]})
  %update1_temp:2 = stream.async.dispatch @ex::@dispatch1(%input[%c0 to %input_size for %input_size]) : (!stream.resource<*>{%input_size}) -> (!stream.resource<*>{%update_size}, !stream.resource<*>{%input_size})
  // CHECK: %[[TARGET2:.+]] = stream.async.dispatch @ex::@dispatch2({{.+}}, %[[TARGET1_TEMP]]#0[%c98304 to %c147456 for %[[UPDATE_SIZE]]]) : ({{.+}}) -> %[[TARGET1_TEMP]]#0{%[[TARGET_SIZE]]}
  %update2 = stream.async.dispatch @ex::@dispatch2(%update1_temp#1[%c0 to %input_size for %input_size]) : (!stream.resource<*>{%input_size}) -> !stream.resource<*>{%update_size}
  // CHECK: %[[TARGET3:.+]] = stream.async.dispatch @ex::@dispatch3({{.+}}, %[[TARGET2]][%c147456 to %c196608 for %[[UPDATE_SIZE]]]) : ({{.+}}) -> %[[TARGET2]]{%[[TARGET_SIZE]]}
  %update3 = stream.async.dispatch @ex::@dispatch3(%input[%c0 to %input_size for %input_size]) : (!stream.resource<*>{%input_size}) -> !stream.resource<*>{%update_size}
  // CHECK-NOT: stream.async.alloca
  %target = stream.async.alloca : !stream.resource<*>{%target_size}
  // CHECK-NOT: stream.async.update
  %target0 = stream.async.update %update0, %target[%c0 to %c49152] : !stream.resource<*>{%update_size} -> %target as !stream.resource<*>{%target_size}
  // CHECK-NOT: stream.async.update
  %target1 = stream.async.update %update1_temp#0, %target0[%c49152 to %c98304] : !stream.resource<*>{%update_size} -> %target0 as !stream.resource<*>{%target_size}
  // CHECK-NOT: stream.async.update
  %target2 = stream.async.update %update2, %target1[%c98304 to %c147456] : !stream.resource<*>{%update_size} -> %target1 as !stream.resource<*>{%target_size}
  // CHECK-NOT: stream.async.update
  %target3 = stream.async.update %update3, %target2[%c147456 to %c196608] : !stream.resource<*>{%update_size} -> %target2 as !stream.resource<*>{%target_size}
  // CHECK: util.return %[[TARGET3]]
  util.return %target3 : !stream.resource<*>
}

// -----

// Tests that multiple results from the same dispatch routing to the same target
// get placed. This can originate from concats where all producers are fused
// into the same dispatch.

// CHECK-LABEL: @emplaceMultiResultDispatchInto
util.func public @emplaceMultiResultDispatchInto(
    // CHECK-SAME: %[[INPUT:arg[0-9]+]]: !stream.resource<*>, %[[INPUT_SIZE:arg[0-9]+]]: index,
    %input: !stream.resource<*>, %input_size: index,
    // CHECK-SAME: %[[UPDATE_SIZE:arg[0-9]+]]: index, %[[TARGET_SIZE:arg[0-9]+]]: index
    %update_size: index, %target_size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  // CHECK: %[[TARGET:.+]] = stream.async.alloca
  // CHECK: %[[DISPATCH:.+]]:2 = stream.async.dispatch @ex::@dispatch0
  // CHECK-SAME: ({{.+}}, %[[TARGET]][%c0 to %c32 for %[[UPDATE_SIZE]]], %[[TARGET]][%c32 to %c64 for %[[UPDATE_SIZE]]]) :
  // CHECK-SAME: ({{.+}}) -> (%[[TARGET]]{%[[TARGET_SIZE]]}, %[[TARGET]]{%[[TARGET_SIZE]]})
  %update:2 = stream.async.dispatch @ex::@dispatch0(%input[%c0 to %input_size for %input_size]) :
      (!stream.resource<*>{%input_size}) -> (!stream.resource<*>{%update_size}, !stream.resource<*>{%update_size})
  // CHECK-NOT: stream.async.alloca
  %target = stream.async.alloca : !stream.resource<*>{%target_size}
  // CHECK-NOT: stream.async.update
  %target0 = stream.async.update %update#0, %target[%c0 to %c32] : !stream.resource<*>{%update_size} -> %target as !stream.resource<*>{%target_size}
  // CHECK-NOT: stream.async.update
  %target1 = stream.async.update %update#1, %target0[%c32 to %c64] : !stream.resource<*>{%update_size} -> %target0 as !stream.resource<*>{%target_size}
  // CHECK: util.return %[[DISPATCH]]#1
  util.return %target1 : !stream.resource<*>
}

// -----

// Test that multiple results inserted out of order get tied back to a dispatch
// consistently.

// CHECK-LABEL: @emplaceMultiResultDispatchIntoSwapped
util.func public @emplaceMultiResultDispatchIntoSwapped(
    // CHECK-SAME: %[[INPUT:arg[0-9]+]]: !stream.resource<*>, %[[INPUT_SIZE:arg[0-9]+]]: index,
    %input: !stream.resource<*>, %input_size: index,
    // CHECK-SAME: %[[UPDATE_SIZE:arg[0-9]+]]: index, %[[TARGET_SIZE:arg[0-9]+]]: index
    %update_size: index, %target_size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  // CHECK: %[[TARGET:.+]] = stream.async.alloca
  // CHECK: %[[DISPATCH:.+]]:2 = stream.async.dispatch @ex::@dispatch0
  // CHECK-SAME: ({{.+}}, %[[TARGET]][%c32 to %c64 for %[[UPDATE_SIZE]]], %[[TARGET]][%c0 to %c32 for %[[UPDATE_SIZE]]]) :
  // CHECK-SAME: ({{.+}}) -> (%[[TARGET]]{%[[TARGET_SIZE]]}, %[[TARGET]]{%[[TARGET_SIZE]]})
  %update:2 = stream.async.dispatch @ex::@dispatch0(%input[%c0 to %input_size for %input_size]) :
      (!stream.resource<*>{%input_size}) -> (!stream.resource<*>{%update_size}, !stream.resource<*>{%update_size})
  // CHECK-NOT: stream.async.alloca
  %target = stream.async.alloca : !stream.resource<*>{%target_size}
  // CHECK-NOT: stream.async.update
  %target0 = stream.async.update %update#1, %target[%c0 to %c32] : !stream.resource<*>{%update_size} -> %target as !stream.resource<*>{%target_size}
  // CHECK-NOT: stream.async.update
  %target1 = stream.async.update %update#0, %target0[%c32 to %c64] : !stream.resource<*>{%update_size} -> %target0 as !stream.resource<*>{%target_size}
  // CHECK: util.return %[[DISPATCH]]#0
  util.return %target1 : !stream.resource<*>
}

// -----

// Tests that we handle sparse emplacement (some results emplaced, some not).
// If any result is emplaced all results get upgraded to tied allocas.

// CHECK-LABEL: @emplaceSparseMultiResult
util.func public @emplaceSparseMultiResult(
    // CHECK-SAME: %[[INPUT:arg[0-9]+]]: !stream.resource<*>, %[[INPUT_SIZE:arg[0-9]+]]: index,
    %input: !stream.resource<*>, %input_size: index,
    // CHECK-SAME: %[[UPDATE_SIZE:arg[0-9]+]]: index, %[[TARGET_SIZE:arg[0-9]+]]: index
    %update_size: index, %target_size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  // CHECK-DAG: %[[UNEMPLACED:.+]] = stream.async.alloca : !stream.resource<*>{%[[UPDATE_SIZE]]}
  // CHECK-DAG: %[[TARGET:.+]] = stream.async.alloca : !stream.resource<*>{%[[TARGET_SIZE]]}
  // CHECK: %[[DISPATCH:.+]]:3 = stream.async.dispatch @ex::@dispatch0
  // CHECK-SAME: (%[[INPUT]][%c0 to %[[INPUT_SIZE]] for %[[INPUT_SIZE]]], %[[TARGET]][%c0 to %c32 for %[[UPDATE_SIZE]]], %[[UNEMPLACED]][%c0 to %[[UPDATE_SIZE]] for %[[UPDATE_SIZE]]], %[[TARGET]][%c32 to %c64 for %[[UPDATE_SIZE]]]) :
  // CHECK-SAME: ({{.+}}) -> (%[[TARGET]]{%[[TARGET_SIZE]]}, %[[UNEMPLACED]]{%[[UPDATE_SIZE]]}, %[[TARGET]]{%[[TARGET_SIZE]]})
  %update:3 = stream.async.dispatch @ex::@dispatch0(%input[%c0 to %input_size for %input_size]) :
      (!stream.resource<*>{%input_size}) -> (!stream.resource<*>{%update_size}, !stream.resource<*>{%update_size}, !stream.resource<*>{%update_size})
  // CHECK-NOT: stream.async.alloca
  %target = stream.async.alloca : !stream.resource<*>{%target_size}
  // CHECK-NOT: stream.async.update
  %target0 = stream.async.update %update#0, %target[%c0 to %c32] : !stream.resource<*>{%update_size} -> %target as !stream.resource<*>{%target_size}
  // CHECK-NOT: stream.async.update
  %target1 = stream.async.update %update#2, %target0[%c32 to %c64] : !stream.resource<*>{%update_size} -> %target0 as !stream.resource<*>{%target_size}
  // CHECK: util.return %[[DISPATCH]]#2
  util.return %target1 : !stream.resource<*>
}

// -----

// Tests that sequences with data dependencies don't hoist beyond them.

// CHECK-LABEL: @emplaceDependentDispatchSequence
util.func public @emplaceDependentDispatchSequence(
    // CHECK-SAME: %[[INPUT:arg[0-9]+]]: !stream.resource<*>, %[[INPUT_SIZE:arg[0-9]+]]: index,
    %input: !stream.resource<*>, %input_size: index,
    // CHECK-SAME: %[[UPDATE_SIZE:arg[0-9]+]]: index, %[[TARGET_SIZE:arg[0-9]+]]: index
    %update_size: index, %target_size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c49152 = arith.constant 49152 : index
  %c98304 = arith.constant 98304 : index
  // CHECK: stream.async.alloca
  // CHECK-NEXT: %[[TARGET0:.+]] = stream.async.dispatch @ex::@dispatch0
  %update0 = stream.async.dispatch @ex::@dispatch0(%input[%c0 to %input_size for %input_size]) : (!stream.resource<*>{%input_size}) -> !stream.resource<*>{%update_size}
  // CHECK-NEXT: %[[UPDATE1_ORIGIN:.+]] = stream.async.dispatch @ex::@dispatch1
  %update1_origin = stream.async.dispatch @ex::@dispatch1(%input[%c0 to %input_size for %input_size]) : (!stream.resource<*>{%input_size}) -> !stream.resource<*>{%update_size}
  // CHECK-NEXT: %[[UPDATE1:.+]] = util.optimization_barrier %[[UPDATE1_ORIGIN]]
  %update1 = util.optimization_barrier %update1_origin : !stream.resource<*>
  // CHECK-NOT: stream.async.alloca
  %target = stream.async.alloca : !stream.resource<*>{%target_size}
  // CHECK-NOT: stream.async.update
  %target0 = stream.async.update %update0, %target[%c0 to %c49152] : !stream.resource<*>{%update_size} -> %target as !stream.resource<*>{%target_size}
  // CHECK-NEXT: %[[TARGET1:.+]] = stream.async.update %[[UPDATE1]], %[[TARGET0]]
  %target1 = stream.async.update %update1, %target0[%c49152 to %c98304] : !stream.resource<*>{%update_size} -> %target0 as !stream.resource<*>{%target_size}
  // CHECK-NEXT: util.return %[[TARGET1]]
  util.return %target1 : !stream.resource<*>
}

// -----

// CHECK-LABEL: @emplaceMultiResultDependentDispatchSequence
util.func public @emplaceMultiResultDependentDispatchSequence(
    // CHECK-SAME: %[[INPUT:arg[0-9]+]]: !stream.resource<*>, %[[INPUT_SIZE:arg[0-9]+]]: index,
    %input: !stream.resource<*>, %input_size: index,
    // CHECK-SAME: %[[UPDATE_SIZE:arg[0-9]+]]: index, %[[TARGET_SIZE:arg[0-9]+]]: index
    %update_size: index, %target_size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c49152 = arith.constant 49152 : index
  // CHECK: %[[UPDATE0:.+]]:2 = stream.async.dispatch @ex::@dispatch0
  %update0:2 = stream.async.dispatch @ex::@dispatch0(%input[%c0 to %input_size for %input_size]) : (!stream.resource<*>{%input_size}) -> (!stream.resource<*>{%update_size}, !stream.resource<*>{%update_size})
  // CHECK-NEXT: %[[UPDATE1:.+]] = stream.async.dispatch @ex::@dispatch1
  %update1 = stream.async.dispatch @ex::@dispatch1(%update0#1[%c0 to %update_size for %input_size]) : (!stream.resource<*>{%update_size}) -> !stream.resource<*>{%update_size}
  // CHECK: %[[TARGET1:.+]] = stream.async.update %[[UPDATE0]]#0, %[[UPDATE1]]
  %target0 = stream.async.update %update0#0, %update1[%c0 to %c49152] : !stream.resource<*>{%update_size} -> %update1_origin as !stream.resource<*>{%target_size}
  // CHECK-NEXT: util.return %[[TARGET1]]
  util.return %target0 : !stream.resource<*>
}

// -----

// Targets that access storage in 4-byte units keep a 6-byte result at byte 6
// out of its neighbour's word.

// CHECK-LABEL: @emplaceNarrowConcatSixBytes
util.func public @emplaceNarrowConcatSixBytes(%input: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> attributes {
  stream.resources = #stream.resource_config<{min_buffer_offset_alignment = 16, min_access_granularity = 4}>
} {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  // CHECK: %[[UPDATE:.+]] = stream.async.dispatch @ex::@dispatch{{.+}} -> !stream.resource<*>{%c6}
  %update = stream.async.dispatch @ex::@dispatch(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update %[[UPDATE]], %{{.+}}[%c6 to %c12]
  %result = stream.async.update %update, %target[%c6 to %c12] : !stream.resource<*>{%c6} -> %target as !stream.resource<*>{%c12}
  util.return %result : !stream.resource<*>
}

// Byte-granular targets keep the placement.

// CHECK-LABEL: @emplaceNarrowConcatByteGranular
util.func public @emplaceNarrowConcatByteGranular(%input: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  // CHECK: stream.async.dispatch @ex::@dispatch{{.+}} -> %{{.+}}{%c12}
  // CHECK-NOT: stream.async.update
  %update = stream.async.dispatch @ex::@dispatch(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  %result = stream.async.update %update, %target[%c6 to %c12] : !stream.resource<*>{%c6} -> %target as !stream.resource<*>{%c12}
  util.return %result : !stream.resource<*>
}

// -----

// Dynamic bounds emplace only when divisibility proves whole access units.

// CHECK-LABEL: @emplaceNarrowDynamic
util.func public @emplaceNarrowDynamic(%input: !stream.resource<*>, %target: !stream.resource<*>, %target_size: index, %offset_arg: index, %size_arg: index) -> (!stream.resource<*>, !stream.resource<*>) attributes {
  stream.resources = #stream.resource_config<{min_buffer_offset_alignment = 16, min_access_granularity = 4}>
} {
  %c0 = arith.constant 0 : index
  %offset = util.assume.int %offset_arg<udiv = 8> : index
  %size = util.assume.int %size_arg<udiv = 4> : index
  %end = arith.addi %offset, %size : index
  // CHECK: %[[ALIGNED:.+]] = stream.async.dispatch @ex::@aligned{{.+}} -> %{{.+}}{%{{.+}}}
  %aligned = stream.async.dispatch @ex::@aligned(%input[%c0 to %size for %size]) : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  %0 = stream.async.update %aligned, %target[%offset to %end] : !stream.resource<*>{%size} -> %target as !stream.resource<*>{%target_size}
  %unproven_end = arith.addi %offset_arg, %size_arg : index
  // CHECK: %[[UNPROVEN:.+]] = stream.async.dispatch @ex::@unproven{{.+}} -> !stream.resource<*>{%{{.+}}}
  %unproven = stream.async.dispatch @ex::@unproven(%input[%c0 to %size_arg for %size_arg]) : (!stream.resource<*>{%size_arg}) -> !stream.resource<*>{%size_arg}
  // CHECK: stream.async.update %[[UNPROVEN]]
  %1 = stream.async.update %unproven, %0[%offset_arg to %unproven_end] : !stream.resource<*>{%size_arg} -> %0 as !stream.resource<*>{%target_size}
  util.return %0, %1 : !stream.resource<*>, !stream.resource<*>
}

// -----

// A result that ends an allocation it owns may end inside a word: the
// allocation's padding is its own. The same end inside an argument is not.

// CHECK-LABEL: @emplaceNarrowOwnedTail
util.func public @emplaceNarrowOwnedTail(%input: !stream.resource<*>, %arg_target: !stream.resource<*>) -> (!stream.resource<*>, !stream.resource<*>) attributes {
  stream.resources = #stream.resource_config<{min_buffer_offset_alignment = 16, min_access_granularity = 4}>
} {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c8 = arith.constant 8 : index
  %c14 = arith.constant 14 : index
  %alloca = stream.async.alloca : !stream.resource<*>{%c14}
  // CHECK: stream.async.dispatch @ex::@owned{{.+}} -> %{{.+}}{%c14}
  %owned = stream.async.dispatch @ex::@owned(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  %0 = stream.async.update %owned, %alloca[%c8 to %c14] : !stream.resource<*>{%c6} -> %alloca as !stream.resource<*>{%c14}
  // CHECK: %[[BORROWED:.+]] = stream.async.dispatch @ex::@borrowed{{.+}} -> !stream.resource<*>{%c6}
  %borrowed = stream.async.dispatch @ex::@borrowed(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update %[[BORROWED]]
  %1 = stream.async.update %borrowed, %arg_target[%c8 to %c14] : !stream.resource<*>{%c6} -> %arg_target as !stream.resource<*>{%c14}
  util.return %0, %1 : !stream.resource<*>, !stream.resource<*>
}

// -----

// A device that may select a word-granular target takes its granularity.

util.global private @device = #hal.device.select<[
  #hal.device.target<"word", {stream.min_access_granularity = 4 : i64}> : !hal.device,
  #hal.device.target<"byte"> : !hal.device
]> : !hal.device

// CHECK-LABEL: @emplaceNarrowSelectedDevice
util.func public @emplaceNarrowSelectedDevice(%input: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  // CHECK: %[[UPDATE:.+]] = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch{{.+}} -> !stream.resource<*>{%c6}
  %update = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update on(#hal.device.affinity<@device>) %[[UPDATE]]
  %result = stream.async.update on(#hal.device.affinity<@device>) %update, %target[%c6 to %c12] : !stream.resource<*>{%c6} -> %target as !stream.resource<*>{%c12}
  util.return %result : !stream.resource<*>
}

// -----

// An optimal set of one device constrains like the device itself.

util.global private @device = #hal.device.select<[
  #hal.device.target<"word", {stream.min_access_granularity = 4 : i64}> : !hal.device,
  #hal.device.target<"byte"> : !hal.device
]> : !hal.device

// CHECK-LABEL: @emplaceNarrowOptimalSingleton
util.func public @emplaceNarrowOptimalSingleton(%input: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  // CHECK: %[[UPDATE:.+]] = stream.async.dispatch on(#hal.device.optimal<[#hal.device.affinity<@device>]>) @ex::@dispatch{{.+}} -> !stream.resource<*>{%c6}
  %update = stream.async.dispatch on(#hal.device.optimal<[#hal.device.affinity<@device>]>) @ex::@dispatch(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update on(#hal.device.optimal<[#hal.device.affinity<@device>]>) %[[UPDATE]]
  %result = stream.async.update on(#hal.device.optimal<[#hal.device.affinity<@device>]>) %update, %target[%c6 to %c12] : !stream.resource<*>{%c6} -> %target as !stream.resource<*>{%c12}
  util.return %result : !stream.resource<*>
}

// -----

// An optimal set takes its most constrained member's unit, in either order;
// whole words still emplace.

util.global private @word = #hal.device.target<"word", {stream.min_access_granularity = 4 : i64}> : !hal.device
util.global private @byte = #hal.device.target<"byte"> : !hal.device

// CHECK-LABEL: @emplaceNarrowOptimalMixed
util.func public @emplaceNarrowOptimalMixed(%input: !stream.resource<*>, %target0: !stream.resource<*>, %target1: !stream.resource<*>, %target2: !stream.resource<*>) -> (!stream.resource<*>, !stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  // CHECK: %[[UPDATE0:.+]] = stream.async.dispatch on(#hal.device.optimal<[#hal.device.affinity<@word>, #hal.device.affinity<@byte>]>) @ex::@dispatch0{{.+}} -> !stream.resource<*>{%c6}
  %update0 = stream.async.dispatch on(#hal.device.optimal<[#hal.device.affinity<@word>, #hal.device.affinity<@byte>]>) @ex::@dispatch0(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update on({{.+}}) %[[UPDATE0]]
  %result0 = stream.async.update on(#hal.device.optimal<[#hal.device.affinity<@word>, #hal.device.affinity<@byte>]>) %update0, %target0[%c6 to %c12] : !stream.resource<*>{%c6} -> %target0 as !stream.resource<*>{%c12}
  // CHECK: %[[UPDATE1:.+]] = stream.async.dispatch on(#hal.device.optimal<[#hal.device.affinity<@byte>, #hal.device.affinity<@word>]>) @ex::@dispatch1{{.+}} -> !stream.resource<*>{%c6}
  %update1 = stream.async.dispatch on(#hal.device.optimal<[#hal.device.affinity<@byte>, #hal.device.affinity<@word>]>) @ex::@dispatch1(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update on({{.+}}) %[[UPDATE1]]
  %result1 = stream.async.update on(#hal.device.optimal<[#hal.device.affinity<@byte>, #hal.device.affinity<@word>]>) %update1, %target1[%c6 to %c12] : !stream.resource<*>{%c6} -> %target1 as !stream.resource<*>{%c12}
  // CHECK: stream.async.dispatch on({{.+}}) @ex::@dispatch2({{.+}}) -> %{{.+}}{%c16}
  // CHECK-NOT: stream.async.update
  %update2 = stream.async.dispatch on(#hal.device.optimal<[#hal.device.affinity<@byte>, #hal.device.affinity<@word>]>) @ex::@dispatch2(%input[%c0 to %c8 for %c8]) : (!stream.resource<*>{%c8}) -> !stream.resource<*>{%c8}
  %result2 = stream.async.update on(#hal.device.optimal<[#hal.device.affinity<@byte>, #hal.device.affinity<@word>]>) %update2, %target2[%c8 to %c16] : !stream.resource<*>{%c8} -> %target2 as !stream.resource<*>{%c16}
  util.return %result0, %result1, %result2 : !stream.resource<*>, !stream.resource<*>, !stream.resource<*>
}

// -----

// A byte-granular optimal set leaves the function's own unit in force.

util.global private @byte = #hal.device.target<"byte"> : !hal.device

// CHECK-LABEL: @emplaceNarrowOptimalExplicitConfig
util.func public @emplaceNarrowOptimalExplicitConfig(%input: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> attributes {
  stream.resources = #stream.resource_config<{min_buffer_offset_alignment = 16, min_access_granularity = 4}>
} {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  // CHECK: %[[UPDATE:.+]] = stream.async.dispatch on({{.+}}) @ex::@dispatch{{.+}} -> !stream.resource<*>{%c6}
  %update = stream.async.dispatch on(#hal.device.optimal<[#hal.device.affinity<@byte>]>) @ex::@dispatch(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update on({{.+}}) %[[UPDATE]]
  %result = stream.async.update on(#hal.device.optimal<[#hal.device.affinity<@byte>]>) %update, %target[%c6 to %c12] : !stream.resource<*>{%c6} -> %target as !stream.resource<*>{%c12}
  util.return %result : !stream.resource<*>
}

// -----

// A subview's end is not its allocation's end, and its offsets are relative
// to the subview; neither proves whole access units.

// CHECK-LABEL: @emplaceNarrowSubviewTail
util.func public @emplaceNarrowSubviewTail(%input: !stream.resource<*>) -> !stream.resource<*> attributes {
  stream.resources = #stream.resource_config<{min_buffer_offset_alignment = 16, min_access_granularity = 4}>
} {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c6 = arith.constant 6 : index
  %c8 = arith.constant 8 : index
  %c14 = arith.constant 14 : index
  %c32 = arith.constant 32 : index
  %alloca = stream.async.alloca : !stream.resource<*>{%c32}
  %subview = stream.resource.subview %alloca[%c2] : !stream.resource<*>{%c32} -> !stream.resource<*>{%c14}
  // CHECK: %[[UPDATE:.+]] = stream.async.dispatch @ex::@dispatch{{.+}} -> !stream.resource<*>{%c6}
  %update = stream.async.dispatch @ex::@dispatch(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update %[[UPDATE]]
  %result = stream.async.update %update, %subview[%c8 to %c14] : !stream.resource<*>{%c6} -> %subview as !stream.resource<*>{%c14}
  util.return %result : !stream.resource<*>
}

// -----

// An empty update touches no access unit.

// CHECK-LABEL: @emplaceNarrowEmpty
util.func public @emplaceNarrowEmpty(%input: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> attributes {
  stream.resources = #stream.resource_config<{min_buffer_offset_alignment = 16, min_access_granularity = 4}>
} {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  // CHECK: stream.async.dispatch @ex::@dispatch{{.+}} -> %{{.+}}{%c12}
  // CHECK-NOT: stream.async.update
  %update = stream.async.dispatch @ex::@dispatch(%input[%c0 to %c0 for %c0]) : (!stream.resource<*>{%c0}) -> !stream.resource<*>{%c0}
  %result = stream.async.update %update, %target[%c6 to %c6] : !stream.resource<*>{%c0} -> %target as !stream.resource<*>{%c12}
  util.return %result : !stream.resource<*>
}

// -----

// A device target that records no granularity keeps byte placement.

util.global private @device = #hal.device.target<"byte"> : !hal.device

// CHECK-LABEL: @emplaceNarrowUnrecordedDevice
util.func public @emplaceNarrowUnrecordedDevice(%input: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  // CHECK: stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch{{.+}} -> %{{.+}}{%c12}
  // CHECK-NOT: stream.async.update
  %update = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  %result = stream.async.update on(#hal.device.affinity<@device>) %update, %target[%c6 to %c12] : !stream.resource<*>{%c6} -> %target as !stream.resource<*>{%c12}
  util.return %result : !stream.resource<*>
}

// -----

// A device that may fall back to a word-granular device takes its
// granularity.

util.global private @primary = #hal.device.target<"word", {stream.min_access_granularity = 4 : i64}> : !hal.device
util.global private @device = #hal.device.select<[
  #hal.device.target<"byte"> : !hal.device,
  #hal.device.fallback<@primary> : !hal.device
]> : !hal.device

// CHECK-LABEL: @emplaceNarrowFallbackDevice
util.func public @emplaceNarrowFallbackDevice(%input: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  // CHECK: %[[UPDATE:.+]] = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch{{.+}} -> !stream.resource<*>{%c6}
  %update = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update on(#hal.device.affinity<@device>) %[[UPDATE]]
  %result = stream.async.update on(#hal.device.affinity<@device>) %update, %target[%c6 to %c12] : !stream.resource<*>{%c6} -> %target as !stream.resource<*>{%c12}
  util.return %result : !stream.resource<*>
}

// -----

// A device global without a resolvable initial value takes the module's
// legacy device targets.

module attributes {hal.device.targets = [#hal.device.target<"word", {stream.min_access_granularity = 4 : i64}> : !hal.device]} {
util.global private @device = #hal.device.ordinal<0> : !hal.device

// CHECK-LABEL: @emplaceNarrowLegacyDevice
util.func public @emplaceNarrowLegacyDevice(%input: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  // CHECK: %[[UPDATE:.+]] = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch{{.+}} -> !stream.resource<*>{%c6}
  %update = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update on(#hal.device.affinity<@device>) %[[UPDATE]]
  %result = stream.async.update on(#hal.device.affinity<@device>) %update, %target[%c6 to %c12] : !stream.resource<*>{%c6} -> %target as !stream.resource<*>{%c12}
  util.return %result : !stream.resource<*>
}
}

// -----

// Cyclic fallbacks do not hide a reachable word-granular device.

module {
util.global private @word = #hal.device.target<"word", {stream.min_access_granularity = 4 : i128}> : !hal.device
util.global private @device = #hal.device.select<[
  #hal.device.fallback<@cycle> : !hal.device,
  #hal.device.fallback<@word> : !hal.device
]> : !hal.device
util.global private @cycle = #hal.device.fallback<@device> : !hal.device

// CHECK-LABEL: @emplaceNarrowCyclicFallbackDevice
util.func public @emplaceNarrowCyclicFallbackDevice(%input: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  // CHECK: %[[UPDATE:.+]] = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch{{.+}} -> !stream.resource<*>{%c6}
  %update = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update on(#hal.device.affinity<@device>) %[[UPDATE]]
  %result = stream.async.update on(#hal.device.affinity<@device>) %update, %target[%c6 to %c12] : !stream.resource<*>{%c6} -> %target as !stream.resource<*>{%c12}
  util.return %result : !stream.resource<*>
}
}

// -----

// Unresolved fallbacks use the legacy constraints instead of asserting.

module attributes {hal.device.targets = [#hal.device.target<"word", {stream.min_access_granularity = 4 : i64}> : !hal.device]} {
util.global private @device = #hal.device.fallback<@unresolved> : !hal.device

// CHECK-LABEL: @emplaceNarrowUnresolvedFallbackDevice
util.func public @emplaceNarrowUnresolvedFallbackDevice(%input: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  // CHECK: %[[UPDATE:.+]] = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch{{.+}} -> !stream.resource<*>{%c6}
  %update = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update on(#hal.device.affinity<@device>) %[[UPDATE]]
  %result = stream.async.update on(#hal.device.affinity<@device>) %update, %target[%c6 to %c12] : !stream.resource<*>{%c6} -> %target as !stream.resource<*>{%c12}
  util.return %result : !stream.resource<*>
}
}

// -----

// An operation override cannot weaken its placement's physical access unit.
util.global private @device = #hal.device.target<"word", {stream.min_access_granularity = 4 : i64}> : !hal.device

// CHECK-LABEL: @emplaceNarrowExplicitCannotWeakenTarget
util.func public @emplaceNarrowExplicitCannotWeakenTarget(%input: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  // CHECK: %[[UPDATE:.+]] = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch{{.+}} -> !stream.resource<*>{%c6}
  %update = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update on(#hal.device.affinity<@device>) %[[UPDATE]]
  %result = stream.async.update on(#hal.device.affinity<@device>) %update, %target[%c6 to %c12] : !stream.resource<*>{%c6} -> %target as !stream.resource<*>{%c12} attributes {
    stream.resources = #stream.resource_config<{min_buffer_offset_alignment = 16, min_access_granularity = 1}>
  }
  util.return %result : !stream.resource<*>
}

// -----

// A scope override cannot weaken a placement inherited from its function.
util.global private @device = #hal.device.target<"word", {stream.min_access_granularity = 4 : i64}> : !hal.device

// CHECK-LABEL: @emplaceNarrowScopeCannotWeakenTarget
util.func public @emplaceNarrowScopeCannotWeakenTarget(%input: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> attributes {stream.affinity = #hal.device.affinity<@device>} {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  // CHECK: %[[UPDATE:.+]] = stream.async.dispatch @ex::@dispatch{{.+}} -> !stream.resource<*>{%c6}
  %update = stream.async.dispatch @ex::@dispatch(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update %[[UPDATE]]
  %result = stream.async.update %update, %target[%c6 to %c12] : !stream.resource<*>{%c6} -> %target as !stream.resource<*>{%c12} attributes {
    stream.resources = #stream.resource_config<{min_buffer_offset_alignment = 16, min_access_granularity = 1}>
  }
  util.return %result : !stream.resource<*>
}

// -----

// Module ownership permits device lookup while independent callable bodies are
// processed in parallel. All three supported callable kinds obey the same unit.
util.global private @device = #hal.device.target<"word", {stream.min_access_granularity = 4 : i64}> : !hal.device

// CHECK-LABEL: @emplaceParallelUtilFunc
util.func public @emplaceParallelUtilFunc(%input: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  // CHECK: %[[UPDATE:.+]] = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch{{.+}} -> !stream.resource<*>{%c6}
  %update = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update on(#hal.device.affinity<@device>) %[[UPDATE]]
  %result = stream.async.update on(#hal.device.affinity<@device>) %update, %target[%c6 to %c12] : !stream.resource<*>{%c6} -> %target as !stream.resource<*>{%c12} attributes {
    stream.resources = #stream.resource_config<{min_buffer_offset_alignment = 16, min_access_granularity = 1}>
  }
  util.return %result : !stream.resource<*>
}

// CHECK-LABEL: @emplaceParallelFuncFunc
func.func @emplaceParallelFuncFunc(%input: !stream.resource<*>, %target: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  // CHECK: %[[UPDATE:.+]] = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch{{.+}} -> !stream.resource<*>{%c6}
  %update = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update on(#hal.device.affinity<@device>) %[[UPDATE]]
  %result = stream.async.update on(#hal.device.affinity<@device>) %update, %target[%c6 to %c12] : !stream.resource<*>{%c6} -> %target as !stream.resource<*>{%c12} attributes {
    stream.resources = #stream.resource_config<{min_buffer_offset_alignment = 16, min_access_granularity = 1}>
  }
  func.return %result : !stream.resource<*>
}

// CHECK-LABEL: util.initializer
util.initializer {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c12 = arith.constant 12 : index
  %input = stream.async.alloca : !stream.resource<*>{%c6}
  %target = stream.async.alloca : !stream.resource<*>{%c12}
  // CHECK: %[[UPDATE:.+]] = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch{{.+}} -> !stream.resource<*>{%c6}
  %update = stream.async.dispatch on(#hal.device.affinity<@device>) @ex::@dispatch(%input[%c0 to %c6 for %c6]) : (!stream.resource<*>{%c6}) -> !stream.resource<*>{%c6}
  // CHECK: stream.async.update on(#hal.device.affinity<@device>) %[[UPDATE]]
  %result = stream.async.update on(#hal.device.affinity<@device>) %update, %target[%c6 to %c12] : !stream.resource<*>{%c6} -> %target as !stream.resource<*>{%c12} attributes {
    stream.resources = #stream.resource_config<{min_buffer_offset_alignment = 16, min_access_granularity = 1}>
  }
  util.return
}

func.func private @external_resource_transform(!stream.resource<*>)
