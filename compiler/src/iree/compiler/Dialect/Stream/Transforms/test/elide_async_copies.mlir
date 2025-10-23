// RUN: iree-opt --split-input-file --iree-stream-elide-async-copies --cse %s | FileCheck %s

// Tests that a normal clone-on-multiple-uses pattern has the last clone elided.
// This is what the --iree-stream-materialize-copy-on-write pass generates and
// expects us to clean up.

// CHECK-LABEL: @multiUseTiedOperand
util.func public @multiUseTiedOperand(%size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  %c789_i32 = arith.constant 789 : i32
  // CHECK: %[[SPLAT:.+]] = stream.async.splat
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[CLONE0:.+]] = stream.async.clone %[[SPLAT]]
  %clone0 = stream.async.clone %splat : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[FILL0:.+]] = stream.async.fill %c456_i32, %[[CLONE0]]
  %fill0 = stream.async.fill %c456_i32, %clone0[%c0 to %c128 for %c128] : i32 -> %1 as !stream.resource<*>{%size}
  // CHECK-NOT: stream.async.clone
  %clone1 = stream.async.clone %splat : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[FILL1:.+]] = stream.async.fill %c789_i32, %[[SPLAT]]
  %fill1 = stream.async.fill %c789_i32, %clone1[%c128 to %c256 for %c128] : i32 -> %3 as !stream.resource<*>{%size}
  // CHECK: util.return %[[FILL0]], %[[FILL1]]
  util.return %fill0, %fill1 : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests a copy of a by-value function argument gets elided.
// Since the caller passes in the last live reference the callee is allowed to
// mutate the memory in-place.

// CHECK-LABEL: @argMoveCallee
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>
util.func private @argMoveCallee(%arg: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %arg : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[ARG0]]
  %fill = stream.async.fill %c123_i32, %clone[%c0 to %c128 for %c128] : i32 -> %0 as !stream.resource<*>{%size}
  // CHECK: util.return %[[FILL]]
  util.return %fill : !stream.resource<*>
}
// CHECK: @argMoveCaller
util.func public @argMoveCaller(%size: index) -> !stream.resource<*> {
  %c123_i32 = arith.constant 123 : i32
  // CHECK: stream.async.splat
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  %result = util.call @argMoveCallee(%splat, %size) : (!stream.resource<*>, index) -> !stream.resource<*>
  util.return %result : !stream.resource<*>
}

// -----

// Tests a copy we cannot elide because the function argument is used after the
// call and passed by const-reference.

// CHECK-LABEL: @argCopyCallee
util.func private @argCopyCallee(%arg: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: stream.async.clone
  %clone = stream.async.clone %arg : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: stream.async.fill
  %fill = stream.async.fill %c123_i32, %clone[%c0 to %c128 for %c128] : i32 -> %0 as !stream.resource<*>{%size}
  util.return %fill : !stream.resource<*>
}
// CHECK: @argCopyCaller
util.func public @argCopyCaller(%size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c123_i32 = arith.constant 123 : i32
  // CHECK: stream.async.splat
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  %result = util.call @argCopyCallee(%splat, %size) : (!stream.resource<*>, index) -> !stream.resource<*>
  util.return %splat, %result : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that block arguments that are chained as last-use will get their
// clones elided while those that are used multiple times will not.
// The first splat is analyzed to be threaded through as the last possible
// use each time meaning that it can be mutated in place. The second splat
// is conditionally chosen to be the initial splat or the new value and as such
// needs to preserve the copy so the original splat is not mutated.

// CHECK-LABEL: @blockArgMove
// CHECK-SAME: (%[[COND:.+]]: i1
util.func private @blockArgMove(%cond: i1, %size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  // CHECK: %[[SPLAT0:.+]] = stream.async.splat %c123
  %splat0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[SPLAT1:.+]] = stream.async.splat %c456
  %splat1 = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: cf.br ^bb1(%[[SPLAT0]], %[[SPLAT1]]
  cf.br ^bb1(%splat0, %splat1 : !stream.resource<*>, !stream.resource<*>)
// CHECK: ^bb1(%[[BB1_ARG0:.+]]: !stream.resource<*>, %[[BB1_ARG1:.+]]: !stream.resource<*>)
^bb1(%bb1_0: !stream.resource<*>, %bb1_1: !stream.resource<*>):
  // CHECK-NOT: stream.async.clone
  %clone0 = stream.async.clone %bb1_0 : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[FILL0:.+]] = stream.async.fill %c123_i32, %[[BB1_ARG0]]
  %fill0 = stream.async.fill %c123_i32, %clone0[%c0 to %c128 for %c128] : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[CLONE1:.+]] = stream.async.clone %[[BB1_ARG1]]
  %clone1 = stream.async.clone %bb1_1 : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[FILL1:.+]] = stream.async.fill %c456_i32, %[[CLONE1]]
  %fill1 = stream.async.fill %c456_i32, %clone1[%c0 to %c128 for %c128] : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[SELECT:.+]] = arith.select %[[COND]], %[[SPLAT1]], %[[FILL1]]
  %bb1_1_new = arith.select %cond, %splat1, %fill1 : !stream.resource<*>
  // CHECK: cf.cond_br %[[COND]], ^bb1(%[[FILL0]], %[[SELECT]]
  // CHECK-SAME:               ^bb2(%[[FILL0]], %[[SELECT]]
  cf.cond_br %cond, ^bb1(%fill0, %bb1_1_new : !stream.resource<*>, !stream.resource<*>),
                 ^bb2(%fill0, %bb1_1_new : !stream.resource<*>, !stream.resource<*>)
^bb2(%bb2_0: !stream.resource<*>, %bb2_1: !stream.resource<*>):
  util.return %bb2_0, %bb2_1 : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that slices aren't elided when there are ops our folding doesn't (yet)
// support.

// CHECK-LABEL: @slice_unsupported_fold
util.func private @slice_unsupported_fold(%producer: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: stream.async.slice
  %slice = stream.async.slice %producer[%c100 to %c200] : !stream.resource<*>{%c300} -> !stream.resource<*>{%c100}
  // CHECK: stream.async.fill
  %consumer = stream.async.fill %c123_i32, %slice[%c0 to %c100 for %c100] : i32 -> !stream.resource<*>{%c100}
  util.return %consumer : !stream.resource<*>
}

// -----

// Tests that slices of tied values don't get folded as our analysis doesn't
// (yet) walk up the use-def chain.

// CHECK-LABEL: @slice_unsupported_tied
util.func private @slice_unsupported_tied(%input: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  %producer_storage = stream.async.alloca : !stream.resource<*>{%c100}
  // CHECK: stream.async.copy
  %producer = stream.async.copy %input[%c0 to %c300], %producer_storage[%c0 to %c300], %c300 : !stream.resource<*>{%c300} -> %producer_storage as !stream.resource<*>{%c300}
  // CHECK: stream.async.slice
  %slice = stream.async.slice %producer[%c100 to %c200] : !stream.resource<*>{%c300} -> !stream.resource<*>{%c100}
  %consumer_storage = stream.async.alloca : !stream.resource<*>{%c100}
  // CHECK: stream.async.copy
  %consumer = stream.async.copy %slice[%c0 to %c100], %consumer_storage[%c0 to %c100], %c300 : !stream.resource<*>{%c100} -> %consumer_storage as !stream.resource<*>{%c100}
  util.return %consumer : !stream.resource<*>
}

// -----

// Tests that sliced ranges that overlap other used ranges don't fold if there
// are writes as the copy is required for correctness.

// CHECK-LABEL: @slice_overlap_preventing
util.func private @slice_overlap_preventing(%producer: !stream.resource<*>) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: stream.async.slice
  %slice = stream.async.slice %producer[%c100 to %c200] : !stream.resource<*>{%c300} -> !stream.resource<*>{%c100}
  %consumer_storage = stream.async.alloca : !stream.resource<*>{%c100}
  // CHECK: stream.async.copy
  %consumer = stream.async.copy %slice[%c0 to %c100], %consumer_storage[%c0 to %c100], %c300 : !stream.resource<*>{%c100} -> %consumer_storage as !stream.resource<*>{%c100}
  // This fill overlaps the sliced range and should block the fold.
  // CHECK: stream.async.fill
  %fill = stream.async.fill %c123_i32, %producer[%c0 to %c200 for %c200] : i32 -> !stream.resource<*>{%c300}
  util.return %consumer, %fill : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that sliced ranges that don't overlap other used ranges fold.

// CHECK-LABEL: @slice_overlap_exclusive
// CHECK-SAME: (%[[PRODUCER:.+]]: !stream.resource<*>)
util.func private @slice_overlap_exclusive(%producer: !stream.resource<*>) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-NOT: stream.async.slice
  %slice = stream.async.slice %producer[%c100 to %c200] : !stream.resource<*>{%c300} -> !stream.resource<*>{%c100}
  %consumer_storage = stream.async.alloca : !stream.resource<*>{%c100}
  // CHECK: stream.async.copy %[[PRODUCER]][%c100 to %c200]
  %consumer = stream.async.copy %slice[%c0 to %c100], %consumer_storage[%c0 to %c100], %c300 : !stream.resource<*>{%c100} -> %consumer_storage as !stream.resource<*>{%c100}
  // CHECK: stream.async.fill
  %fill = stream.async.fill %c123_i32, %producer[%c200 to %c300 for %c100] : i32 -> !stream.resource<*>{%c300}
  util.return %consumer, %fill : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that sliced ranges that overlap but just for reads.

// CHECK-LABEL: @slice_overlap_readonly
// CHECK-SAME: (%[[PRODUCER:.+]]: !stream.resource<*>)
util.func private @slice_overlap_readonly(%producer: !stream.resource<*>) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c101 = arith.constant 101 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  // CHECK-NOT: stream.async.slice
  %slice = stream.async.slice %producer[%c100 to %c200] : !stream.resource<*>{%c300} -> !stream.resource<*>{%c100}
  %consumer_storage_0 = stream.async.alloca : !stream.resource<*>{%c100}
  // CHECK: stream.async.copy %[[PRODUCER]][%c100 to %c200]
  %consumer_0 = stream.async.copy %slice[%c0 to %c100], %consumer_storage_0[%c0 to %c100], %c300 : !stream.resource<*>{%c100} -> %consumer_storage as !stream.resource<*>{%c100}
  %consumer_storage_1 = stream.async.alloca : !stream.resource<*>{%c100}
  // CHECK: stream.async.copy %[[PRODUCER]][%c101 to %c201]
  %consumer_1 = stream.async.copy %slice[%c1 to %c101], %consumer_storage_1[%c0 to %c100], %c300 : !stream.resource<*>{%c100} -> %consumer_storage as !stream.resource<*>{%c100}
  util.return %consumer_0, %consumer_1 : !stream.resource<*>, !stream.resource<*>
}

// -----

stream.executable private @ex {
  stream.executable.export public @dispatch workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @slice_dispatch_fold
// CHECK-SAME: (%[[PRODUCER:.+]]: !stream.resource<*>)
util.func private @slice_dispatch_fold(%producer: !stream.resource<*>) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c30 = arith.constant 30 : index
  %c100 = arith.constant 100 : index
  %c200 = arith.constant 200 : index
  %c300 = arith.constant 300 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-NOT: stream.async.slice
  %slice = stream.async.slice %producer[%c100 to %c200] : !stream.resource<*>{%c300} -> !stream.resource<*>{%c100}
  // CHECK: stream.async.dispatch @ex::@dispatch(%c123_i32, %[[PRODUCER]][%c110 to %c130 for %c20]) : (i32, !stream.resource<*>{%c300}) -> !stream.resource<*>{%c100}
  %consumer = stream.async.dispatch @ex::@dispatch(%c123_i32, %slice[%c10 to %c30 for %c20]) : (i32, !stream.resource<*>{%c100}) -> !stream.resource<*>{%c100}
  util.return %consumer : !stream.resource<*>
}

// -----

// Tests that clones performing lifetime conversions are NOT elided.
// When a clone changes the resource lifetime (e.g., * -> variable, * -> external),
// it must be preserved even if it's the last use of the source.
// This is critical for correctness when the specific lifetime is required by
// consumers like util.global.store or function boundaries.

util.global private mutable @global_var : !stream.resource<variable>

stream.executable private @dispatch {
  stream.executable.export public @entry workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
  builtin.module {
    func.func @entry() {
      return
    }
  }
}

// CHECK-LABEL: @cloneToVariable
util.func public @cloneToVariable() -> !stream.resource<variable> {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32

  // Dispatch produces a generic resource.
  // CHECK: %[[DISPATCH:.+]] = stream.async.dispatch
  %dispatch = stream.async.dispatch @dispatch::@entry() : () -> !stream.resource<*>{%c4}

  // Clone to variable lifetime for storing in global.
  // This clone MUST NOT be elided even though dispatch is only used here,
  // because it changes lifetime from * to variable.
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[DISPATCH]]
  // CHECK-SAME: !stream.resource<*>{%c4} -> !stream.resource<variable>{%c4}
  %clone = stream.async.clone %dispatch : !stream.resource<*>{%c4} -> !stream.resource<variable>{%c4}

  // Global store requires exact variable type.
  // CHECK: util.global.store %[[CLONE]], @global_var
  util.global.store %clone, @global_var : !stream.resource<variable>

  util.return %clone : !stream.resource<variable>
}

// -----

// Tests that clones to external lifetime are preserved.
// External resources cross module boundaries and require the specific lifetime.

// CHECK-LABEL: @cloneToExternal
util.func public @cloneToExternal() -> !stream.resource<external> {
  %c4 = arith.constant 4 : index
  %c123_i32 = arith.constant 123 : i32

  // Splat produces a generic resource.
  // CHECK: %[[SPLAT:.+]] = stream.async.splat
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c4}

  // Clone to external lifetime for export.
  // This clone MUST NOT be elided even though splat is only used here,
  // because it changes lifetime from * to external.
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[SPLAT]]
  // CHECK-SAME: !stream.resource<*>{%c4} -> !stream.resource<external>{%c4}
  %clone = stream.async.clone %splat : !stream.resource<*>{%c4} -> !stream.resource<external>{%c4}

  util.return %clone : !stream.resource<external>
}

// -----

// Tests that clones with the same lifetime CAN still be elided (baseline
// behavior). This ensures we don't break existing optimizations.

// CHECK-LABEL: @cloneSameLifetimePreserved
util.func public @cloneSameLifetimePreserved() -> !stream.resource<*> {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32

  // CHECK: %[[SPLAT:.+]] = stream.async.splat %c123
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c4}

  // Clone with same lifetime (* -> *) and last use CAN be elided.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %splat : !stream.resource<*>{%c4} -> !stream.resource<*>{%c4}

  // CHECK: stream.async.fill %c456_i32, %[[SPLAT]]
  %fill = stream.async.fill %c456_i32, %clone[%c0 to %c4 for %c4] : i32 -> %0 as !stream.resource<*>{%c4}

  util.return %fill : !stream.resource<*>
}

// -----

// Tests aliasing pattern where user provides output storage.
// The dispatch reuses the input buffer as output (tied operand) - clone should be elided.

stream.executable private @ex0 {
  stream.executable.export public @dispatch workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @aliasing_user_provided_output_callee
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index)
util.func private @aliasing_user_provided_output_callee(%resource: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // Defensive clone as inserted by MaterializeCopyOnWritePass.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %resource : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // Dispatch reuses input as output (tied operand pattern).
  // CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex0::@dispatch[%c1, %c1, %c1](%[[RESOURCE]][%c0 to %[[SIZE]] for %[[SIZE]]], %[[RESOURCE]][%c0 to %[[SIZE]] for %[[SIZE]]]) : (!stream.resource<*>{%[[SIZE]]}, !stream.resource<*>{%[[SIZE]]}) -> %[[RESOURCE]]{%[[SIZE]]}
  %result = stream.async.dispatch @ex0::@dispatch[%c1, %c1, %c1](%clone[%c0 to %size for %size], %clone[%c0 to %size for %size]) : (!stream.resource<*>{%size}, !stream.resource<*>{%size}) -> %clone{%size}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// CHECK-LABEL: @aliasing_user_provided_output_caller
util.func public @aliasing_user_provided_output_caller(%size: index) -> !stream.resource<*> {
  %c123_i32 = arith.constant 123 : i32
  %initial = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  %result = util.call @aliasing_user_provided_output_callee(%initial, %size) : (!stream.resource<*>, index) -> !stream.resource<*>
  util.return %result : !stream.resource<*>
}

// -----

// Tests dispatch taking same SSA value multiple times (aliasing inputs).
// Clone should be elided since all uses are read-only.

stream.executable private @ex1 {
  stream.executable.export public @dispatch_same_input_twice workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @dispatch_same_value_twice_callee
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index)
util.func private @dispatch_same_value_twice_callee(%resource: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // Defensive clone as inserted by MaterializeCopyOnWritePass.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %resource : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // Dispatch reads the same value twice - both are read-only.
  // CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex1::@dispatch_same_input_twice[%c1, %c1, %c1](%[[RESOURCE]][%c0 to %[[SIZE]] for %[[SIZE]]], %[[RESOURCE]][%c0 to %[[SIZE]] for %[[SIZE]]]) : (!stream.resource<*>{%[[SIZE]]}, !stream.resource<*>{%[[SIZE]]}) -> !stream.resource<*>{%[[SIZE]]}
  %result = stream.async.dispatch @ex1::@dispatch_same_input_twice[%c1, %c1, %c1](%clone[%c0 to %size for %size], %clone[%c0 to %size for %size]) : (!stream.resource<*>{%size}, !stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// CHECK-LABEL: @dispatch_same_value_twice_caller
util.func public @dispatch_same_value_twice_caller(%size: index) -> !stream.resource<*> {
  %c123_i32 = arith.constant 123 : i32
  %initial = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  %result = util.call @dispatch_same_value_twice_callee(%initial, %size) : (!stream.resource<*>, index) -> !stream.resource<*>
  util.return %result : !stream.resource<*>
}

// -----

// Tests splat followed by in-place dispatch (common zero-copy pattern).
// Splat creates buffer, dispatch mutates it in-place without clone.

stream.executable private @ex2 {
  stream.executable.export public @dispatch_inplace workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @splat_then_inplace_dispatch_callee
util.func private @splat_then_inplace_dispatch_callee(%size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SPLAT:.+]] = stream.async.splat %c123_i32
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // Defensive clone as inserted by MaterializeCopyOnWritePass.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %splat : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // Dispatch mutates splat buffer in-place (tied operand).
  // CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex2::@dispatch_inplace[%c1, %c1, %c1](%[[SPLAT]][%c0 to %[[SIZE:.+]] for %[[SIZE]]]) : (!stream.resource<*>{%[[SIZE]]}) -> %[[SPLAT]]{%[[SIZE]]}
  %result = stream.async.dispatch @ex2::@dispatch_inplace[%c1, %c1, %c1](%clone[%c0 to %size for %size]) : (!stream.resource<*>{%size}) -> %clone{%size}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// CHECK-LABEL: @splat_then_inplace_dispatch_caller
util.func public @splat_then_inplace_dispatch_caller(%size: index) -> !stream.resource<*> {
  %result = util.call @splat_then_inplace_dispatch_callee(%size) : (index) -> !stream.resource<*>
  util.return %result : !stream.resource<*>
}

// -----

// Tests splat followed by non-in-place dispatch (creates new output).
// Clone should be elided since splat is last use before dispatch.

stream.executable private @ex3 {
  stream.executable.export public @dispatch_new_output workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @splat_then_new_output_dispatch_callee
util.func private @splat_then_new_output_dispatch_callee(%size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SPLAT:.+]] = stream.async.splat %c123_i32
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // Defensive clone as inserted by MaterializeCopyOnWritePass.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %splat : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // Dispatch reads splat and produces new output (not tied).
  // CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex3::@dispatch_new_output[%c1, %c1, %c1](%[[SPLAT]][%c0 to %[[SIZE:.+]] for %[[SIZE]]]) : (!stream.resource<*>{%[[SIZE]]}) -> !stream.resource<*>{%[[SIZE]]}
  %result = stream.async.dispatch @ex3::@dispatch_new_output[%c1, %c1, %c1](%clone[%c0 to %size for %size]) : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}

// CHECK-LABEL: @splat_then_new_output_dispatch_caller
util.func public @splat_then_new_output_dispatch_caller(%size: index) -> !stream.resource<*> {
  %result = util.call @splat_then_new_output_dispatch_callee(%size) : (index) -> !stream.resource<*>
  util.return %result : !stream.resource<*>
}

// -----

// Tests chained dispatches with user-provided storage (common pattern).
// First dispatch in-place, second dispatch reuses same buffer.

stream.executable private @ex4 {
  stream.executable.export public @dispatch_first workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

stream.executable private @ex5 {
  stream.executable.export public @dispatch_second workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @chained_inplace_dispatches_callee
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index)
util.func private @chained_inplace_dispatches_callee(%resource: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // First defensive clone.
  // CHECK-NOT: stream.async.clone
  %clone1 = stream.async.clone %resource : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // First dispatch mutates in-place.
  // CHECK: %[[DISPATCH1:.+]] = stream.async.dispatch @ex4::@dispatch_first[%c1, %c1, %c1](%[[RESOURCE]][%c0 to %[[SIZE]] for %[[SIZE]]]) : (!stream.resource<*>{%[[SIZE]]}) -> %[[RESOURCE]]{%[[SIZE]]}
  %dispatch1 = stream.async.dispatch @ex4::@dispatch_first[%c1, %c1, %c1](%clone1[%c0 to %size for %size]) : (!stream.resource<*>{%size}) -> %clone1{%size}
  // Second defensive clone.
  // CHECK-NOT: stream.async.clone
  %clone2 = stream.async.clone %dispatch1 : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // Second dispatch also mutates in-place.
  // CHECK: %[[DISPATCH2:.+]] = stream.async.dispatch @ex5::@dispatch_second[%c1, %c1, %c1](%[[DISPATCH1]][%c0 to %[[SIZE]] for %[[SIZE]]]) : (!stream.resource<*>{%[[SIZE]]}) -> %[[DISPATCH1]]{%[[SIZE]]}
  %dispatch2 = stream.async.dispatch @ex5::@dispatch_second[%c1, %c1, %c1](%clone2[%c0 to %size for %size]) : (!stream.resource<*>{%size}) -> %clone2{%size}
  // CHECK: util.return %[[DISPATCH2]]
  util.return %dispatch2 : !stream.resource<*>
}

// CHECK-LABEL: @chained_inplace_dispatches_caller
util.func public @chained_inplace_dispatches_caller(%size: index) -> !stream.resource<*> {
  %c123_i32 = arith.constant 123 : i32
  %initial = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  %result = util.call @chained_inplace_dispatches_callee(%initial, %size) : (!stream.resource<*>, index) -> !stream.resource<*>
  util.return %result : !stream.resource<*>
}
