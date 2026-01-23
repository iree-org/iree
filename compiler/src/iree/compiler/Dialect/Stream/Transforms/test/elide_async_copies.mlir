// RUN: iree-opt --split-input-file --iree-stream-elide-async-copies --cse %s | FileCheck %s

// Tests that clones with the same lifetime can be elided (baseline behavior).

// CHECK-LABEL: @cloneSameLifetime
util.func public @cloneSameLifetime() -> !stream.resource<*> {
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

// CHECK-LABEL: util.func private @argMoveCallee
// CHECK-SAME: (%[[ARG:.+]]: !stream.resource<*>, %{{.+}}: index)
util.func private @argMoveCallee(%arg: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %arg : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[ARG]]
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

// CHECK-LABEL: util.func private @argCopyCallee
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

// CHECK-LABEL: util.func private @blockArgMove
// CHECK-SAME: (%[[COND:.+]]: i1, %{{.+}}: index)
util.func private @blockArgMove(%cond: i1, %size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  // CHECK: %[[SPLAT0:.+]] = stream.async.splat %c123
  %splat0 = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[SPLAT1:.+]] = stream.async.splat %c456
  %splat1 = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: cf.br ^bb1(%[[SPLAT0]], %[[SPLAT1]] : !stream.resource<*>, !stream.resource<*>)
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
  // CHECK: %[[BB1_1_NEW:.+]] = arith.select %[[COND]], %[[SPLAT1]], %[[FILL1]]
  %bb1_1_new = arith.select %cond, %splat1, %fill1 : !stream.resource<*>
  // CHECK: cf.cond_br %[[COND]], ^bb1(%[[FILL0]], %[[BB1_1_NEW]] : !stream.resource<*>, !stream.resource<*>),
  // CHECK-SAME:               ^bb2(%[[FILL0]], %[[BB1_1_NEW]] : !stream.resource<*>, !stream.resource<*>)
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
  // CHECK-DAG: %[[SLICE_START:.+]] = arith.constant 100 : index
  // CHECK-DAG: %[[SLICE_END:.+]] = arith.constant 200 : index
  // CHECK: stream.async.copy %[[PRODUCER]][%[[SLICE_START]] to %[[SLICE_END]]]
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
  %slice = stream.async.slice %producer[%c100 to %c200] : !stream.resource<*>{%c300} -> !stream.resource<*>{%c100}
  %consumer_storage_0 = stream.async.alloca : !stream.resource<*>{%c100}
  // First copy reads slice[0:100] which maps to producer[100:200].
  // CHECK-DAG: %[[COPY0_START:.+]] = arith.constant 100 : index
  // CHECK-DAG: %[[COPY0_END:.+]] = arith.constant 200 : index
  // CHECK: stream.async.copy %[[PRODUCER]][%[[COPY0_START]] to %[[COPY0_END]]]
  // CHECK-NOT: stream.async.slice
  %consumer_0 = stream.async.copy %slice[%c0 to %c100], %consumer_storage_0[%c0 to %c100], %c300 : !stream.resource<*>{%c100} -> %consumer_storage as !stream.resource<*>{%c100}
  %consumer_storage_1 = stream.async.alloca : !stream.resource<*>{%c100}
  // Second copy reads slice[1:101] which maps to producer[101:201].
  // CHECK-DAG: %[[COPY1_START:.+]] = arith.constant 101 : index
  // CHECK-DAG: %[[COPY1_END:.+]] = arith.constant 201 : index
  // CHECK: stream.async.copy %[[PRODUCER]][%[[COPY1_START]] to %[[COPY1_END]]]
  %consumer_1 = stream.async.copy %slice[%c1 to %c101], %consumer_storage_1[%c0 to %c100], %c300 : !stream.resource<*>{%c100} -> %consumer_storage as !stream.resource<*>{%c100}
  util.return %consumer_0, %consumer_1 : !stream.resource<*>, !stream.resource<*>
}

// -----

stream.executable private @ex {
  stream.executable.export public @dispatch workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
  stream.executable.export public @dispatch1 workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
  stream.executable.export public @dispatch2 workgroups() -> (index, index, index) {
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
  %slice = stream.async.slice %producer[%c100 to %c200] : !stream.resource<*>{%c300} -> !stream.resource<*>{%c100}
  // CHECK: stream.async.dispatch @ex::@dispatch(%c123_i32, %[[PRODUCER]][%c110 to %c130 for %c20]) : (i32, !stream.resource<*>{%c300}) -> !stream.resource<*>{%c100}
  %consumer = stream.async.dispatch @ex::@dispatch(%c123_i32, %slice[%c10 to %c30 for %c20]) : (i32, !stream.resource<*>{%c100}) -> !stream.resource<*>{%c100}
  // CHECK-NOT: stream.async.slice
  // CHECK: util.return
  util.return %consumer : !stream.resource<*>
}

// -----

// Tests that clones performing lifetime conversions are NOT elided.
// When a clone changes the resource lifetime (e.g., * -> variable, * -> external),
// it must be preserved even if it's the last use of the source.
// This is critical for correctness when the specific lifetime is required by
// consumers like util.global.store or function boundaries.

util.global private mutable @global_var : !stream.resource<variable>

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

  // CHECK: util.return %[[CLONE]]
  util.return %clone : !stream.resource<external>
}

// -----

// Tests that constant->constant clones are always elided because constants
// are immutable and can be safely aliased even with multiple users.
// This is the key optimization for reducing copies of constant parameters.

// CHECK-LABEL: @constantCloneSingleUse
util.func public @constantCloneSingleUse(%size: index) -> !stream.resource<constant> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index

  // CHECK: %[[CONST:.+]] = stream.async.constant
  %const = stream.async.constant : !stream.resource<constant>{%size} = dense<123> : tensor<128xi32>

  // Constant->constant clone with single use should be elided.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %const : !stream.resource<constant>{%size} -> !stream.resource<constant>{%size}

  // CHECK: util.return %[[CONST]]
  util.return %clone : !stream.resource<constant>
}

// -----

// Tests that constant->constant clones with multiple uses are elided.
// Constants are immutable so aliasing is always safe regardless of use count.
// This pattern appears when constants are used in both initializers and
// model functions.

// CHECK-LABEL: @constantCloneMultiUse
util.func public @constantCloneMultiUse(%size: index) -> (!stream.resource<constant>, !stream.resource<constant>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[CONST:.+]] = stream.async.constant
  %const = stream.async.constant : !stream.resource<constant>{%size} = dense<1.0> : tensor<256xf32>

  // First clone - even though const has multiple users, this should be elided
  // because constants are immutable.
  // CHECK-NOT: stream.async.clone
  %clone0 = stream.async.clone %const : !stream.resource<constant>{%size} -> !stream.resource<constant>{%size}

  // Second clone - also should be elided.
  // CHECK-NOT: stream.async.clone
  %clone1 = stream.async.clone %const : !stream.resource<constant>{%size} -> !stream.resource<constant>{%size}

  // Both dispatches read from the same constant (no mutation).
  // CSE will deduplicate these identical dispatches into one.
  // CHECK: %[[RESULT0:.+]] = stream.async.dispatch @constantDispatch::@entry
  // CHECK-SAME: %[[CONST]]
  %result0 = stream.async.dispatch @constantDispatch::@entry(%clone0[%c0 to %size for %size]) :
      (!stream.resource<constant>{%size}) -> !stream.resource<constant>{%size}

  %result1 = stream.async.dispatch @constantDispatch::@entry(%clone1[%c0 to %size for %size]) :
      (!stream.resource<constant>{%size}) -> !stream.resource<constant>{%size}

  // CHECK: util.return %[[RESULT0]], %[[RESULT0]]
  util.return %result0, %result1 : !stream.resource<constant>, !stream.resource<constant>
}

// -----

// Tests that constant->* clones are still preserved (lifetime conversion).
// This ensures our constant->constant optimization doesn't break the
// existing lifetime conversion logic.

// CHECK-LABEL: @constantToGenericPreserved
util.func public @constantToGenericPreserved(%size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c123_i32 = arith.constant 123 : i32

  // CHECK: %[[CONST:.+]] = stream.async.constant
  %const = stream.async.constant : !stream.resource<constant>{%c4} = dense<123> : tensor<1xi32>

  // Clone from constant to generic lifetime must be preserved.
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[CONST]]
  // CHECK-SAME: !stream.resource<constant>{%c4} -> !stream.resource<*>{%c4}
  %clone = stream.async.clone %const : !stream.resource<constant>{%c4} -> !stream.resource<*>{%c4}

  // CHECK: stream.async.fill %c123_i32, %[[CLONE]]
  %fill = stream.async.fill %c123_i32, %clone[%c0 to %c4 for %c4] : i32 -> %0 as !stream.resource<*>{%c4}

  util.return %fill : !stream.resource<*>
}

// -----

// Tests that constant->constant clones are preserved when the result is tied
// (mutated). This is critical for correctness during initialization where
// constants may be mutated to produce their final values.

// CHECK-LABEL: @constantCloneTiedMutation
util.func public @constantCloneTiedMutation(%size: index) -> !stream.resource<constant> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c456_i32 = arith.constant 456 : i32

  // CHECK: %[[CONST:.+]] = stream.async.constant
  %const = stream.async.constant : !stream.resource<constant>{%size} = dense<123> : tensor<128xi32>

  // Clone before mutation - this clone MUST be preserved because the result
  // is tied to an operation (will be mutated).
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[CONST]]
  %clone = stream.async.clone %const : !stream.resource<constant>{%size} -> !stream.resource<constant>{%size}

  // Fill operation that mutates the clone (tied operand).
  // CHECK: %[[FILLED:.+]] = stream.async.fill %c456_i32, %[[CLONE]]
  %filled = stream.async.fill %c456_i32, %clone[%c0 to %c128 for %c128] : i32 -> %clone as !stream.resource<constant>{%size}

  // CHECK: util.return %[[FILLED]]
  util.return %filled : !stream.resource<constant>
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

// CHECK-LABEL: util.func private @aliasing_user_provided_output_callee
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

// CHECK-LABEL: util.func private @dispatch_same_value_twice_callee
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

// CHECK-LABEL: util.func private @splat_then_inplace_dispatch_callee
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

// CHECK-LABEL: util.func private @splat_then_new_output_dispatch_callee
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

// CHECK-LABEL: util.func private @chained_inplace_dispatches_callee
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

// -----

//===----------------------------------------------------------------------===//
// Inter-procedural by-val/by-ref argument semantics
//===----------------------------------------------------------------------===//

// Tests that a callee called with mixed semantics (both by-val and by-ref)
// must preserve defensive copies since it can't assume semantics.

stream.executable private @ex_mixed {
  stream.executable.export public @dispatch workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: util.func private @mixedSemanticsSameCallee
// CHECK-SAME: (%[[ARG:[^:]+]]: !stream.resource<external>)
util.func private @mixedSemanticsSameCallee(%arg: !stream.resource<external>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // Clone must be preserved because callee is called with both semantics.
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[ARG]]
  %clone = stream.async.clone %arg : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: stream.async.fill %c123_i32, %[[CLONE]]
  %filled = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %clone as !stream.resource<external>{%c100}
  util.return
}

// CHECK-LABEL: @mixedSemanticsByvalCaller
util.func public @mixedSemanticsByvalCaller() {
  %c0_i32 = arith.constant 0 : i32
  %c100 = arith.constant 100 : index
  %resource = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c100}
  // Last use - by-value semantics.
  util.call @mixedSemanticsSameCallee(%resource) : (!stream.resource<external>) -> ()
  util.return
}

// CHECK-LABEL: @mixedSemanticsByrefCaller
util.func public @mixedSemanticsByrefCaller() {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c100 = arith.constant 100 : index
  %resource = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c100}
  // Not last use - by-ref semantics.
  util.call @mixedSemanticsSameCallee(%resource) : (!stream.resource<external>) -> ()
  // Additional use after call.
  %dispatch = stream.async.dispatch @ex_mixed::@dispatch(%resource[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  util.return
}

// -----

//===----------------------------------------------------------------------===//
// Same-type transfer elision
//===----------------------------------------------------------------------===//

// Tests that same-type transfers (external->external) are always elided
// because they're no-ops regardless of by-val/by-ref semantics.

// CHECK-LABEL: @transferSameTypeFromArg
// CHECK-SAME: (%[[ARG:[^:]+]]: !stream.resource<external>)
util.func public @transferSameTypeFromArg(%arg: !stream.resource<external>) -> !stream.resource<external> {
  %c100 = arith.constant 100 : index
  // Same-type transfer is always a no-op and can be elided.
  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %arg : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: util.return %[[ARG]]
  util.return %transfer : !stream.resource<external>
}

// -----

// Tests that type-changing transfers (transient->external) are preserved
// because they change mutability semantics.

// CHECK-LABEL: @transferTypeChangingFromArg
// CHECK-SAME: (%[[ARG:[^:]+]]: !stream.resource<transient>)
util.func public @transferTypeChangingFromArg(%arg: !stream.resource<transient>) -> !stream.resource<external> {
  %c100 = arith.constant 100 : index
  // Type-changing transfer must be preserved.
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[ARG]]
  %transfer = stream.async.transfer %arg : !stream.resource<transient>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<external>
}

// -----

// Tests that transfer chains with type changes are preserved.
// Since the source is a by-ref argument and transfers change types,
// both transfers must be preserved.

// CHECK-LABEL: @transferChainFromArg
// CHECK-SAME: (%[[ARG:[^:]+]]: !stream.resource<constant>)
util.func public @transferChainFromArg(%arg: !stream.resource<constant>) -> !stream.resource<external> {
  %c100 = arith.constant 100 : index
  // CHECK: %[[T1:.+]] = stream.async.transfer %[[ARG]] : !stream.resource<constant>{{.+}} -> !stream.resource<transient>
  %t1 = stream.async.transfer %arg : !stream.resource<constant>{%c100} -> !stream.resource<transient>{%c100}
  // CHECK: %[[T2:.+]] = stream.async.transfer %[[T1]] : !stream.resource<transient>{{.+}} -> !stream.resource<external>
  %t2 = stream.async.transfer %t1 : !stream.resource<transient>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: util.return %[[T2]]
  util.return %t2 : !stream.resource<external>
}

// -----

// Tests that same-type transfers with multiple uses of the source are elided.
// Since same-type transfer is a no-op, both dispatches use the original arg.

stream.executable private @ex_transfer {
  stream.executable.export public @dispatch1 workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
  stream.executable.export public @dispatch2 workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @transferSameTypeMultiUse
// CHECK-SAME: (%[[ARG:[^:]+]]: !stream.resource<external>)
util.func public @transferSameTypeMultiUse(%arg: !stream.resource<external>) -> (!stream.resource<external>, !stream.resource<external>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %arg : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // Both original and transfer result used - transfer still elided.
  // CHECK: %[[D1:.+]] = stream.async.dispatch @ex_transfer::@dispatch1(%[[ARG]][%c0 to %c100 for %c100])
  %dispatch1 = stream.async.dispatch @ex_transfer::@dispatch1(%arg[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  // CHECK: %[[D2:.+]] = stream.async.dispatch @ex_transfer::@dispatch2(%[[ARG]][%c0 to %c100 for %c100])
  %dispatch2 = stream.async.dispatch @ex_transfer::@dispatch2(%transfer[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  // CHECK: util.return %[[D1]], %[[D2]]
  util.return %dispatch1, %dispatch2 : !stream.resource<external>, !stream.resource<external>
}

// -----

//===----------------------------------------------------------------------===//
// Clone from by-ref argument to tied operation
//===----------------------------------------------------------------------===//

// Tests that a clone from a by-ref argument feeding a tied operation must be
// preserved to prevent mutation of the caller's value.

stream.executable private @ex_byref {
  stream.executable.export public @dispatch workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: util.func public @cloneByrefToTied
// CHECK-SAME: (%[[ARG:[^:]+]]: !stream.resource<external>)
util.func public @cloneByrefToTied(%arg: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // Clone must be preserved - arg is used again after call (by-ref semantics).
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[ARG]]
  %clone = stream.async.clone %arg : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // Tied op will mutate - must not mutate the original arg.
  // CHECK: %[[FILLED:.+]] = stream.async.fill %c123_i32, %[[CLONE]]
  %filled = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %clone as !stream.resource<external>{%c100}
  // CHECK: util.return %[[FILLED]]
  util.return %filled : !stream.resource<external>
}

// CHECK-LABEL: @cloneByrefToTiedCaller
util.func public @cloneByrefToTiedCaller() {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c100 = arith.constant 100 : index
  %resource = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c100}
  // Not last use - by-ref.
  %result = util.call @cloneByrefToTied(%resource) : (!stream.resource<external>) -> !stream.resource<external>
  // Original resource used again.
  %dispatch = stream.async.dispatch @ex_byref::@dispatch(%resource[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  util.return
}

// -----

//===----------------------------------------------------------------------===//
// AsyncUpdateOp elision tests
//===----------------------------------------------------------------------===//

// Tests that an update followed by a write-only operation to the same region
// can be elided because the mutation is overwritten.

stream.executable private @ex_update {
  stream.executable.export public @dispatch workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @updateElisionOverwrite
util.func private @updateElisionOverwrite(%size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  // CHECK: %[[TARGET:.+]] = stream.async.splat %c123
  %target = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[UPDATE:.+]] = stream.async.splat %c456
  %update = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%c4}
  // Update writes to [0, 4).
  // CHECK: stream.async.update %[[UPDATE]], %[[TARGET]]
  %updated = stream.async.update %update, %target[%c0 to %c4] : !stream.resource<*>{%c4} -> %target as !stream.resource<*>{%size}
  // The update result is used so we cannot elide it.
  // CHECK: util.return
  util.return %updated : !stream.resource<*>
}

// -----

// Tests that an update where the result is only used for another write
// to the same region can have the update elided.

// CHECK-LABEL: @updateElisionNoRead
util.func private @updateElisionNoRead(%size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  %c789_i32 = arith.constant 789 : i32
  // CHECK: %[[TARGET:.+]] = stream.async.splat %c123
  %target = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[UPDATE:.+]] = stream.async.splat %c456
  %update = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%c4}
  // Update writes to [0, 4).
  // Next user writes to [0, 4), completely overwriting this update.
  // CHECK-NOT: stream.async.update %[[UPDATE]]
  %updated = stream.async.update %update, %target[%c0 to %c4] : !stream.resource<*>{%c4} -> %target as !stream.resource<*>{%size}
  // CHECK: %[[FILL:.+]] = stream.async.fill %c789_i32, %[[TARGET]]
  %filled = stream.async.fill %c789_i32, %updated[%c0 to %c4 for %c4] : i32 -> !stream.resource<*>{%size}
  util.return %filled : !stream.resource<*>
}

// -----

// Tests that update cannot be elided when the updated region is read.

stream.executable private @ex_read {
  stream.executable.export public @dispatch workgroups(%arg0: index, %arg1: index, %arg2: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @updateNotElided_regionRead
util.func private @updateNotElided_regionRead(%size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  // CHECK: %[[TARGET:.+]] = stream.async.splat %c123
  %target = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[UPDATE:.+]] = stream.async.splat %c456
  %update = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%c4}
  // Update writes to [0, 4).
  // CHECK: %[[UPDATED:.+]] = stream.async.update %[[UPDATE]], %[[TARGET]]
  %updated = stream.async.update %update, %target[%c0 to %c4] : !stream.resource<*>{%c4} -> %target as !stream.resource<*>{%size}
  // Dispatch reads from [0, 4) - overlaps with the update, so cannot elide.
  // CHECK: stream.async.dispatch @ex_read::@dispatch[%c1, %c1, %c1](%[[UPDATED]]
  %dispatch = stream.async.dispatch @ex_read::@dispatch[%c1, %c1, %c1](%updated[%c0 to %c4 for %c4]) : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  util.return %dispatch : !stream.resource<*>
}

// -----

// Tests that update cannot be elided when target is a by-ref function argument.

// CHECK-LABEL: util.func private @updateNotElided_byRefArg
util.func private @updateNotElided_byRefArg(%target: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c123_i32 = arith.constant 123 : i32
  %update = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c4}
  // Update on a by-ref arg cannot be elided as mutation might be observable.
  // CHECK: stream.async.update
  %updated = stream.async.update %update, %target[%c0 to %c4] : !stream.resource<*>{%c4} -> %target as !stream.resource<*>{%size}
  util.return %updated : !stream.resource<*>
}

// CHECK-LABEL: @updateNotElided_byRefArg_caller
util.func public @updateNotElided_byRefArg_caller(%size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c123_i32 = arith.constant 123 : i32
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  %result = util.call @updateNotElided_byRefArg(%splat, %size) : (!stream.resource<*>, index) -> !stream.resource<*>
  // Splat is used after call, so arg is by-ref.
  util.return %splat, %result : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that a full-buffer update (where update size == target size) with
// a last-use target can be optimized to replace with source.

// CHECK-LABEL: @updateFullBuffer
util.func private @updateFullBuffer() -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  // CHECK: %[[TARGET:.+]] = stream.async.splat %c123
  %target = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c4}
  // CHECK: %[[UPDATE:.+]] = stream.async.splat %c456
  %update = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%c4}
  // Full buffer update with same sizes - can replace result with source.
  // The update itself will remain since it's the only use, but the result
  // usage pattern is what we're testing.
  %updated = stream.async.update %update, %target[%c0 to %c4] : !stream.resource<*>{%c4} -> %target as !stream.resource<*>{%c4}
  // CHECK: util.return
  util.return %updated : !stream.resource<*>
}

// -----

// Tests that an update cannot be elided when its result is used by a tied
// dispatch operation. The tied dispatch passes the buffer through, and
// downstream operations reading from the dispatch result may access the
// update region through the tied result alias.

stream.executable private @ex_tied_passthrough {
  stream.executable.export public @dispatch workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @updateNotElided_tiedDispatch
util.func private @updateNotElided_tiedDispatch(%size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  // CHECK: %[[TARGET:.+]] = stream.async.splat %c123
  %target = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[UPDATE_SRC:.+]] = stream.async.splat %c456
  %update_src = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%c4}
  // Update writes to [0, 4).
  // CHECK: %[[UPDATED:.+]] = stream.async.update %[[UPDATE_SRC]], %[[TARGET]][%c0 to %c4]
  %updated = stream.async.update %update_src, %target[%c0 to %c4] : !stream.resource<*>{%c4} -> %target as !stream.resource<*>{%size}
  // Tied dispatch reads from [4, 8) - disjoint from update region.
  // But output is tied to input, so dispatch result aliases entire buffer.
  // CHECK: %[[DISPATCH:.+]] = stream.async.dispatch @ex_tied_passthrough::@dispatch(%[[UPDATED]][%c4 to %c8 for %c4]) : (!stream.resource<*>{%[[SIZE:.+]]}) -> %[[UPDATED]]{%[[SIZE]]}
  %dispatch = stream.async.dispatch @ex_tied_passthrough::@dispatch(%updated[%c4 to %c8 for %c4]) : (!stream.resource<*>{%size}) -> %updated{%size}
  // Downstream copy reads entire buffer [0, 8) from dispatch result.
  // This read covers the update region [0, 4), so update cannot be elided.
  %output = stream.async.alloca : !stream.resource<*>{%c8}
  // CHECK: stream.async.copy %[[DISPATCH]]
  %copy = stream.async.copy %dispatch[%c0 to %c8], %output[%c0 to %c8], %c8 : !stream.resource<*>{%size} -> %output as !stream.resource<*>{%c8}
  util.return %copy : !stream.resource<*>
}

// -----

// Tests that chained updates (where one update's result is used as target of
// another update) are not incorrectly elided. The first update writes to a
// subset of the buffer, and even though only the second update's result is
// directly used by the export, the first update's write is still visible
// through the exported buffer.
//
// This is the concat pattern: alloca -> update[0:1] -> update[1:2] -> export
// where both updates contribute to the final buffer.

// CHECK-LABEL: @updateChained
util.func private @updateChained() -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c1_i8 = arith.constant 1 : i8
  %c2_i8 = arith.constant 2 : i8
  // CHECK: %[[ALLOCA:.+]] = stream.async.alloca
  %alloca = stream.async.alloca : !stream.resource<*>{%c2}
  // CHECK: %[[SPLAT0:.+]] = stream.async.splat %c1_i8
  %splat0 = stream.async.splat %c1_i8 : i8 -> !stream.resource<*>{%c1}
  // CHECK: %[[SPLAT1:.+]] = stream.async.splat %c2_i8
  %splat1 = stream.async.splat %c2_i8 : i8 -> !stream.resource<*>{%c1}
  // First update writes [1] at position 0. This must NOT be elided because
  // the result is used as target of the second update.
  // CHECK: %[[UPDATE0:.+]] = stream.async.update %[[SPLAT0]], %[[ALLOCA]][%c0 to %c1]
  %update0 = stream.async.update %splat0, %alloca[%c0 to %c1] : !stream.resource<*>{%c1} -> %alloca as !stream.resource<*>{%c2}
  // Second update writes [2] at position 1.
  // CHECK: %[[UPDATE1:.+]] = stream.async.update %[[SPLAT1]], %[[UPDATE0]][%c1 to %c2]
  %update1 = stream.async.update %splat1, %update0[%c1 to %c2] : !stream.resource<*>{%c1} -> %update0 as !stream.resource<*>{%c2}
  // CHECK: util.return %[[UPDATE1]]
  util.return %update1 : !stream.resource<*>
}

// -----

// Tests that an update through a tied chain from a by-ref argument is not
// elided. The target is the result of a fill operation that aliases the
// by-ref input, so the update's mutation is observable to the caller.

// CHECK-LABEL: util.func private @updateNotElided_tiedChainFromByRef
// CHECK-SAME: (%[[INPUT:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index)
util.func private @updateNotElided_tiedChainFromByRef(
    %input: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c123_i32 = arith.constant 123 : i32
  // Fill creates a tied result that aliases input (by-ref arg).
  // CHECK: %[[FILLED:.+]] = stream.async.fill %c123_i32, %[[INPUT]]
  %filled = stream.async.fill %c123_i32, %input[%c0 to %c4 for %c4]
      : i32 -> %input as !stream.resource<*>{%size}
  // CHECK: %[[UPDATE_SRC:.+]] = stream.async.splat %c123_i32
  %update_src = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c4}
  // Update target is %filled, which traces back to by-ref %input.
  // The update must NOT be elided because mutation is observable to caller.
  // CHECK: %[[UPDATED:.+]] = stream.async.update %[[UPDATE_SRC]], %[[FILLED]][%c0 to %c4]
  %updated = stream.async.update %update_src, %filled[%c0 to %c4]
      : !stream.resource<*>{%c4} -> %filled as !stream.resource<*>{%size}
  // CHECK: util.return %[[UPDATED]]
  util.return %updated : !stream.resource<*>
}

// CHECK-LABEL: @updateNotElided_tiedChainFromByRef_caller
util.func public @updateNotElided_tiedChainFromByRef_caller(%size: index)
    -> (!stream.resource<*>, !stream.resource<*>) {
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SPLAT:.+]] = stream.async.splat
  %splat = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[RESULT:.+]] = util.call @updateNotElided_tiedChainFromByRef(%[[SPLAT]]
  %result = util.call @updateNotElided_tiedChainFromByRef(%splat, %size)
      : (!stream.resource<*>, index) -> !stream.resource<*>
  // Splat used after call = by-ref semantics.
  // CHECK: util.return %[[SPLAT]], %[[RESULT]]
  util.return %splat, %result : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that an update cannot be elided when target has multiple users.
// Even if the update itself would be safe to elide, we conservatively keep it
// when target is used elsewhere to avoid subtle aliasing issues.

// CHECK-LABEL: @updateNotElided_targetMultipleUsers
// CHECK-SAME: (%[[SIZE:.+]]: index)
util.func private @updateNotElided_targetMultipleUsers(%size: index)
    -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: %[[TARGET:.+]] = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%[[SIZE]]}
  %target = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[UPDATE_SRC:.+]] = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%[[C4]]}
  %update_src = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%c4}
  // Target has two users: the update and the return.
  // The update must NOT be elided because target has multiple users.
  // CHECK: %[[UPDATED:.+]] = stream.async.update %[[UPDATE_SRC]], %[[TARGET]][%[[C0]] to %[[C4]]]
  %updated = stream.async.update %update_src, %target[%c0 to %c4]
      : !stream.resource<*>{%c4} -> %target as !stream.resource<*>{%size}
  // Return both target and updated - target has multiple users.
  // CHECK: util.return %[[TARGET]], %[[UPDATED]]
  util.return %target, %updated : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that chained updates to the SAME region can have the first elided.
// When the second update completely overwrites the same range as the first,
// the first update is dead and can be removed.

// CHECK-LABEL: @updateElided_chainedSameRegion
// CHECK-SAME: (%[[SIZE:.+]]: index)
util.func private @updateElided_chainedSameRegion(%size: index)
    -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  %c789_i32 = arith.constant 789 : i32
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK: %[[TARGET:.+]] = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%[[SIZE]]}
  %target = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // First splat for update1 - remains but its update is elided.
  %update1_src = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%c4}
  // CHECK: %[[UPDATE2_SRC:.+]] = stream.async.splat %c789_i32 : i32 -> !stream.resource<*>{%[[C4]]}
  %update2_src = stream.async.splat %c789_i32 : i32 -> !stream.resource<*>{%c4}
  // First update writes [0, 4). This CAN be elided because the second update
  // writes to the exact same range, fully overwriting it.
  %updated1 = stream.async.update %update1_src, %target[%c0 to %c4]
      : !stream.resource<*>{%c4} -> %target as !stream.resource<*>{%size}
  // Second update writes [0, 4) - exact same region, overwrites first.
  // The first update was elided, so this now updates %target directly.
  // CHECK: %[[UPDATED2:.+]] = stream.async.update %[[UPDATE2_SRC]], %[[TARGET]][%[[C0]] to %[[C4]]]
  %updated2 = stream.async.update %update2_src, %updated1[%c0 to %c4]
      : !stream.resource<*>{%c4} -> %updated1 as !stream.resource<*>{%size}
  // CHECK: util.return %[[UPDATED2]]
  util.return %updated2 : !stream.resource<*>
}

// -----

// Tests that chained updates to DIFFERENT regions preserve both updates.
// When the second update writes to a different range, the first update's
// write is still visible through the final result.

// CHECK-LABEL: @updateNotElided_chainedDifferentRegion
// CHECK-SAME: (%[[SIZE:.+]]: index)
util.func private @updateNotElided_chainedDifferentRegion(%size: index)
    -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  %c789_i32 = arith.constant 789 : i32
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
  // CHECK: %[[TARGET:.+]] = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%[[SIZE]]}
  %target = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[UPDATE1_SRC:.+]] = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%[[C4]]}
  %update1_src = stream.async.splat %c456_i32 : i32 -> !stream.resource<*>{%c4}
  // CHECK: %[[UPDATE2_SRC:.+]] = stream.async.splat %c789_i32 : i32 -> !stream.resource<*>{%[[C4]]}
  %update2_src = stream.async.splat %c789_i32 : i32 -> !stream.resource<*>{%c4}
  // First update writes [0, 4). This must NOT be elided because second
  // update writes to a different region [4, 8).
  // CHECK: %[[UPDATED1:.+]] = stream.async.update %[[UPDATE1_SRC]], %[[TARGET]][%[[C0]] to %[[C4]]]
  %updated1 = stream.async.update %update1_src, %target[%c0 to %c4]
      : !stream.resource<*>{%c4} -> %target as !stream.resource<*>{%size}
  // Second update writes [4, 8) - different region, first must be kept.
  // CHECK: %[[UPDATED2:.+]] = stream.async.update %[[UPDATE2_SRC]], %[[UPDATED1]][%[[C4]] to %[[C8]]]
  %updated2 = stream.async.update %update2_src, %updated1[%c4 to %c8]
      : !stream.resource<*>{%c4} -> %updated1 as !stream.resource<*>{%size}
  // CHECK: util.return %[[UPDATED2]]
  util.return %updated2 : !stream.resource<*>
}

// -----

// Tests that an update followed by a copy to a DIFFERENT region preserves the
// update. This pattern occurs in tensor.concat lowering where:
//   1. An output buffer is allocated
//   2. First concat operand is written via update to [0, N)
//   3. Second concat operand is written via copy to [N, M)
// The copy's result aliases the entire target buffer, so downstream reads of
// regions [0, N) must see the update's data.

// CHECK-LABEL: @updateNotElided_followedByCopyToDifferentRegion
// CHECK-SAME: (%[[SIZE:.+]]: index, %[[SRC:.+]]: !stream.resource<*>)
util.func private @updateNotElided_followedByCopyToDifferentRegion(
    %size: index, %copy_src: !stream.resource<*>)
    -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
  // Allocate output buffer.
  // CHECK: %[[ALLOCA:.+]] = stream.async.alloca
  %alloca = stream.async.alloca : !stream.resource<*>{%size}
  // Create data to update into first region.
  // CHECK: %[[UPDATE_SRC:.+]] = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%[[C4]]}
  %update_src = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c4}
  // Update writes [0, 4). This must NOT be elided because the copy's result
  // aliases the entire buffer and downstream operations may read [0, 4).
  // CHECK: %[[UPDATED:.+]] = stream.async.update %[[UPDATE_SRC]], %[[ALLOCA]][%[[C0]] to %[[C4]]]
  %updated = stream.async.update %update_src, %alloca[%c0 to %c4]
      : !stream.resource<*>{%c4} -> %alloca as !stream.resource<*>{%size}
  // Copy writes [4, 8) - different region than update. The copy result aliases
  // the full [0, 8) buffer, so reads of [0, 4) from copy result must see
  // the update's data.
  // CHECK: %[[COPIED:.+]] = stream.async.copy %[[SRC]][%[[C0]] to %[[C4]]], %[[UPDATED]][%[[C4]] to %[[C8]]], %[[C4]]
  %copied = stream.async.copy %copy_src[%c0 to %c4], %updated[%c4 to %c8], %c4
      : !stream.resource<*>{%size} -> %updated as !stream.resource<*>{%size}
  // CHECK: util.return %[[COPIED]]
  util.return %copied : !stream.resource<*>
}

// -----

// Tests that an update followed by a copy to the SAME region allows elision.
// When the copy fully overwrites the update region, the update's data is never
// observable and can be safely elided.

// CHECK-LABEL: @updateElided_followedByCopyToSameRegion
// CHECK-SAME: (%[[SIZE:.+]]: index, %[[SRC:.+]]: !stream.resource<*>)
util.func private @updateElided_followedByCopyToSameRegion(
    %size: index, %copy_src: !stream.resource<*>)
    -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // Allocate output buffer.
  // CHECK: %[[ALLOCA:.+]] = stream.async.alloca
  %alloca = stream.async.alloca : !stream.resource<*>{%size}
  // Create data to update into first region.
  %update_src = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c4}
  // Update writes [0, 4). This CAN be elided because the copy writes to the
  // exact same region, fully overwriting the update's data.
  // CHECK-NOT: stream.async.update
  %updated = stream.async.update %update_src, %alloca[%c0 to %c4]
      : !stream.resource<*>{%c4} -> %alloca as !stream.resource<*>{%size}
  // Copy writes [0, 4) - same region as update, fully overwrites.
  // CHECK: %[[COPIED:.+]] = stream.async.copy %[[SRC]][%[[C0]] to %[[C4]]], %[[ALLOCA]][%[[C0]] to %[[C4]]], %[[C4]]
  %copied = stream.async.copy %copy_src[%c0 to %c4], %updated[%c0 to %c4], %c4
      : !stream.resource<*>{%size} -> %updated as !stream.resource<*>{%size}
  // CHECK: util.return %[[COPIED]]
  util.return %copied : !stream.resource<*>
}

// -----

// Tests that an update followed by a tied dispatch preserves the update.
// Unlike copy operations which are write-only to the target, dispatch
// operations have read-write semantics (they read the tied operand before
// writing). This means the dispatch would read the update's data, making
// elision unsafe regardless of whether the regions match.

stream.executable private @ex_dispatch {
  stream.executable.export public @dispatch workgroups(%arg0: index) -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @updateNotElided_followedByTiedDispatch
// CHECK-SAME: (%[[SIZE:.+]]: index)
util.func private @updateNotElided_followedByTiedDispatch(
    %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // Allocate output buffer.
  // CHECK: %[[ALLOCA:.+]] = stream.async.alloca
  %alloca = stream.async.alloca : !stream.resource<*>{%size}
  // Create data to update into first region.
  // CHECK: %[[UPDATE_SRC:.+]] = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%[[C4]]}
  %update_src = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c4}
  // Update writes [0, 4). This must NOT be elided because the dispatch reads
  // the tied operand (read-write semantics) before writing to it.
  // CHECK: %[[UPDATED:.+]] = stream.async.update %[[UPDATE_SRC]], %[[ALLOCA]][%[[C0]] to %[[C4]]]
  %updated = stream.async.update %update_src, %alloca[%c0 to %c4]
      : !stream.resource<*>{%c4} -> %alloca as !stream.resource<*>{%size}
  // Dispatch has read-write semantics on tied operand.
  // CHECK: %[[DISPATCH:.+]] = stream.async.dispatch @ex_dispatch::@dispatch[%c1](%[[UPDATED]][%[[C0]] to %[[C4]] for %[[C4]]]) : (!stream.resource<*>{%[[SIZE]]}) -> %[[UPDATED]]{%[[SIZE]]}
  %dispatch = stream.async.dispatch @ex_dispatch::@dispatch[%c1](%updated[%c0 to %c4 for %c4]) : (!stream.resource<*>{%size}) -> %updated{%size}
  // CHECK: util.return %[[DISPATCH]]
  util.return %dispatch : !stream.resource<*>
}

// -----

// Tests that multiple clones of an immutable source are all elided.
// When neither source nor clone results are mutated, all clones can be elided
// because sharing the underlying buffer has no observable effect.
// Uses a splat (op result) as source since function args have by-ref semantics.
// Note: After clone elision, CSE may fold identical dispatches.

stream.executable private @dispatch_ex {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%in: !stream.binding, %out: !stream.binding) {
      return
    }
  }
}

// CHECK-LABEL: @multiCloneImmutableSource
util.func public @multiCloneImmutableSource(%size: index) -> (!stream.resource<*>, !stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  // Source is a splat (op result, not function arg).
  // CHECK: %[[SOURCE:.+]] = stream.async.splat
  %source = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // Multiple clones of same source, all used by read-only dispatches.
  // All clones can be elided because neither source nor results are mutated.
  // After elision, CSE folds the identical dispatches into one.
  // CHECK-NOT: stream.async.clone
  %clone0 = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  %d0 = stream.async.dispatch @dispatch_ex::@dispatch(%clone0[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  %clone1 = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  %d1 = stream.async.dispatch @dispatch_ex::@dispatch(%clone1[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  %clone2 = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  %d2 = stream.async.dispatch @dispatch_ex::@dispatch(%clone2[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // CHECK: %[[DISPATCH:.+]] = stream.async.dispatch @dispatch_ex::@dispatch(%[[SOURCE]]
  // CHECK: util.return %[[DISPATCH]], %[[DISPATCH]], %[[DISPATCH]]
  util.return %d0, %d1, %d2 : !stream.resource<*>, !stream.resource<*>, !stream.resource<*>
}

// -----

// Missed optimization: Copy writes to [0, 16) which fully contains the update
// region [4, 8), so the update could theoretically be elided. However, we
// currently only check for exact range matches, not supersets. This is
// conservative (safe) behavior - improving this would require integer range
// analysis to prove containment.
// CHECK-LABEL: @updateNotElided_copyWritesSupersetRegion
// CHECK-SAME: (%[[SIZE:.+]]: index)
util.func private @updateNotElided_copyWritesSupersetRegion(
    %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
  // CHECK-DAG: %[[C4:.+]] = arith.constant 4 : index
  // CHECK-DAG: %[[C8:.+]] = arith.constant 8 : index
  // CHECK-DAG: %[[C16:.+]] = arith.constant 16 : index
  // Allocate output buffer.
  // CHECK: %[[ALLOCA:.+]] = stream.async.alloca
  %alloca = stream.async.alloca : !stream.resource<*>{%size}
  // Create data to update into subregion [4, 8).
  // CHECK: %[[UPDATE_SRC:.+]] = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%[[C4]]}
  %update_src = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c4}
  // Update writes to [4, 8). Not elided because copy writes superset [0, 16).
  // CHECK: %[[UPDATED:.+]] = stream.async.update %[[UPDATE_SRC]], %[[ALLOCA]][%[[C4]] to %[[C8]]]
  %updated = stream.async.update %update_src, %alloca[%c4 to %c8]
      : !stream.resource<*>{%c4} -> %alloca as !stream.resource<*>{%size}
  // Create source for copy that writes superset region [0, 16).
  // CHECK: %[[COPY_SRC:.+]] = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%[[C16]]}
  %copy_src = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%c16}
  // Copy writes [0, 16) which fully contains [4, 8) but we don't detect this.
  // CHECK: %[[COPY:.+]] = stream.async.copy %[[COPY_SRC]][%[[C0]] to %[[C16]]], %[[UPDATED]][%[[C0]] to %[[C16]]], %[[C16]]
  %copy = stream.async.copy %copy_src[%c0 to %c16], %updated[%c0 to %c16], %c16
      : !stream.resource<*>{%c16} -> %updated as !stream.resource<*>{%size}
  // CHECK: util.return %[[COPY]]
  util.return %copy : !stream.resource<*>
}

// -----

// Tests that multiple clones of an immutable source are all elided.
// When neither source nor clone results are mutated, all clones can be elided
// because sharing the underlying buffer has no observable effect.
// Uses a splat (op result) as source since function args have by-ref semantics.
// Note: After clone elision, CSE may fold identical dispatches.

stream.executable private @dispatch_ex {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%in: !stream.binding, %out: !stream.binding) {
      return
    }
  }
}

// CHECK-LABEL: @multiCloneImmutableSource
util.func public @multiCloneImmutableSource(%size: index) -> (!stream.resource<*>, !stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  // Source is a splat (op result, not function arg).
  // CHECK: %[[SOURCE:.+]] = stream.async.splat
  %source = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // Multiple clones of same source, all used by read-only dispatches.
  // All clones can be elided because neither source nor results are mutated.
  // After elision, CSE folds the identical dispatches into one.
  // CHECK-NOT: stream.async.clone
  %clone0 = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  %d0 = stream.async.dispatch @dispatch_ex::@dispatch(%clone0[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  %clone1 = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  %d1 = stream.async.dispatch @dispatch_ex::@dispatch(%clone1[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  %clone2 = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  %d2 = stream.async.dispatch @dispatch_ex::@dispatch(%clone2[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // CHECK: %[[DISPATCH:.+]] = stream.async.dispatch @dispatch_ex::@dispatch(%[[SOURCE]]
  // CHECK: util.return %[[DISPATCH]], %[[DISPATCH]], %[[DISPATCH]]
  util.return %d0, %d1, %d2 : !stream.resource<*>, !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that clone is preserved when result is mutated AND clone is not the
// last user of source. This ensures we don't incorrectly elide when mutation
// could affect other users of the source.

// CHECK-LABEL: @cloneMutatedNotLastUser
util.func public @cloneMutatedNotLastUser(%size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SOURCE:.+]] = stream.async.splat
  %source = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // Clone is NOT the last user because source is used by d1 after the clone.
  // Clone result is mutated by tied dispatch, so clone MUST be preserved.
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[SOURCE]]
  %clone = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[D0:.+]] = stream.async.dispatch @dispatch_ex::@dispatch(%[[CLONE]]{{.*}}) : ({{.*}}) -> %[[CLONE]]
  %d0 = stream.async.dispatch @dispatch_ex::@dispatch(%clone[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> %clone{%size}
  // Source used after clone - clone is not the last user.
  // CHECK: %[[D1:.+]] = stream.async.dispatch @dispatch_ex::@dispatch(%[[SOURCE]]
  %d1 = stream.async.dispatch @dispatch_ex::@dispatch(%source[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // CHECK: util.return %[[D0]], %[[D1]]
  util.return %d0, %d1 : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that clone is elided when result is mutated BUT clone IS the last user
// of source. The last-user optimization safely elides the clone because
// source's lifetime ends at the clone point anyway.

// CHECK-LABEL: @cloneMutatedLastUser
util.func public @cloneMutatedLastUser(%size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SOURCE:.+]] = stream.async.splat
  %source = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // Clone IS the last (and only) user of source.
  // Even though result is mutated, elision is safe because source is dead after.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[D:.+]] = stream.async.dispatch @dispatch_ex::@dispatch(%[[SOURCE]]{{.*}}) : ({{.*}}) -> %[[SOURCE]]
  %d = stream.async.dispatch @dispatch_ex::@dispatch(%clone[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> %clone{%size}
  // CHECK: util.return %[[D]]
  util.return %d : !stream.resource<*>
}

// -----

// Tests that clone is preserved when the source is mutated by another operation.
// Even if clone result is not mutated, the source mutation means aliasing would
// expose the mutation to clone users.

// CHECK-LABEL: @clonePreservedSourceMutatedElsewhere
util.func public @clonePreservedSourceMutatedElsewhere(%size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SOURCE:.+]] = stream.async.splat
  %source = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // Clone must be preserved because source is mutated after.
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[SOURCE]]
  %clone = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // Source is mutated (fill writes to it in-place).
  // CHECK: %[[MUTATED_SOURCE:.+]] = stream.async.fill {{.*}}, %[[SOURCE]]
  %mutated_source = stream.async.fill %c123_i32, %source[%c0 to %size for %size] : i32 -> %0 as !stream.resource<*>{%size}
  // Clone is only read (not mutated), but must be preserved because source changed.
  // CHECK: %[[READ_CLONE:.+]] = stream.async.dispatch @dispatch_ex::@dispatch(%[[CLONE]]
  %read_clone = stream.async.dispatch @dispatch_ex::@dispatch(%clone[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // CHECK: util.return %[[MUTATED_SOURCE]], %[[READ_CLONE]]
  util.return %mutated_source, %read_clone : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that clone is preserved when source is mutated via tied dispatch.
// Similar to above but mutation is through dispatch tied operand.

// CHECK-LABEL: @clonePreservedSourceMutatedViaTiedDispatch
util.func public @clonePreservedSourceMutatedViaTiedDispatch(%size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SOURCE:.+]] = stream.async.splat
  %source = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // Clone must be preserved.
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[SOURCE]]
  %clone = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // Source is mutated via tied dispatch result.
  // CHECK: %[[MUTATED:.+]] = stream.async.dispatch @dispatch_ex::@dispatch(%[[SOURCE]]{{.*}}) : ({{.*}}) -> %[[SOURCE]]
  %mutated = stream.async.dispatch @dispatch_ex::@dispatch(%source[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> %source{%size}
  // Clone is read-only.
  // CHECK: %[[READ:.+]] = stream.async.dispatch @dispatch_ex::@dispatch(%[[CLONE]]
  %read = stream.async.dispatch @dispatch_ex::@dispatch(%clone[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // CHECK: util.return %[[MUTATED]], %[[READ]]
  util.return %mutated, %read : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that clone elision respects cross-function mutation.
// If clone is passed to a callee that mutates it, elision must not happen
// if source is used after the call.

// Callee that mutates its argument via tied result.
// CHECK-LABEL: util.func private @mutating_callee
util.func private @mutating_callee(%arg: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  %mutated = stream.async.fill %c123_i32, %arg[%c0 to %size for %size] : i32 -> %0 as !stream.resource<*>{%size}
  util.return %mutated : !stream.resource<*>
}

// CHECK-LABEL: @clonePreservedCrossFunctionMutation
util.func public @clonePreservedCrossFunctionMutation(%size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SOURCE:.+]] = stream.async.splat
  %source = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // Clone must be preserved - callee mutates it.
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[SOURCE]]
  %clone = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // Call mutates the clone.
  // CHECK: %[[CALL_RESULT:.+]] = util.call @mutating_callee(%[[CLONE]]
  %call_result = util.call @mutating_callee(%clone, %size) : (!stream.resource<*>, index) -> !stream.resource<*>
  // Source used after - would see mutation if clone was elided.
  // CHECK: %[[READ:.+]] = stream.async.dispatch @dispatch_ex::@dispatch(%[[SOURCE]]
  %read = stream.async.dispatch @dispatch_ex::@dispatch(%source[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // CHECK: util.return %[[CALL_RESULT]], %[[READ]]
  util.return %call_result, %read : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that clone CAN be elided when callee only reads (no mutation).

// Callee that only reads its argument.
// CHECK-LABEL: util.func private @reading_callee
util.func private @reading_callee(%arg: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // No tied result - just reads and produces new output.
  %result = stream.async.dispatch @dispatch_ex::@dispatch(%arg[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  util.return %result : !stream.resource<*>
}

// CHECK-LABEL: @cloneElidedCrossFunctionReadOnly
util.func public @cloneElidedCrossFunctionReadOnly(%size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SOURCE:.+]] = stream.async.splat
  %source = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // Clone can be elided - callee only reads.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[CALL_RESULT:.+]] = util.call @reading_callee(%[[SOURCE]]
  %call_result = util.call @reading_callee(%clone, %size) : (!stream.resource<*>, index) -> !stream.resource<*>
  // Source used after - safe because callee only read.
  // CHECK: %[[READ:.+]] = stream.async.dispatch @dispatch_ex::@dispatch(%[[SOURCE]]
  %read = stream.async.dispatch @dispatch_ex::@dispatch(%source[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // CHECK: util.return %[[CALL_RESULT]], %[[READ]]
  util.return %call_result, %read : !stream.resource<*>, !stream.resource<*>
}

// -----

// TODO(benvanik): Tests slice aliasing. The clone of source is elided because
// source appears unmutated (slice mutations are tracked separately). This is
// safe because slices have CoW semantics and become independent allocations,
// but tracking slice aliasing could enable better elision decisions.

// CHECK-LABEL: @clonePreservedAfterSliceMutation
util.func public @clonePreservedAfterSliceMutation(%size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c50 = arith.constant 50 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SOURCE:.+]] = stream.async.splat
  %source = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[SLICE:.+]] = stream.async.slice %[[SOURCE]]
  %slice = stream.async.slice %source[%c0 to %c50] : !stream.resource<*>{%size} -> !stream.resource<*>{%c50}
  // Clone elided; slice has CoW semantics so this is safe.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: stream.async.fill {{.*}}, %[[SLICE]]
  %mutated_slice = stream.async.fill %c123_i32, %slice[%c0 to %c50 for %c50] : i32 -> %0 as !stream.resource<*>{%c50}
  // Dispatch uses source directly; slice becomes independent allocation.
  // CHECK: stream.async.dispatch @dispatch_ex::@dispatch(%[[SOURCE]]
  %read = stream.async.dispatch @dispatch_ex::@dispatch(%clone[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  util.return %mutated_slice, %read : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that a cross-lifetime clone (external → *) is preserved by the first
// ElideAsyncCopies pass even when neither side is mutated. Cross-lifetime
// elision is handled by ResourceUsageAnalysis propagating source usage through
// the clone, RefineUsage unifying the types, and the second ElideAsyncCopies
// pass eliding the now-same-type clone.

stream.executable private @dispatch_ex {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%in: !stream.binding, %out: !stream.binding) {
      return
    }
  }
}

// CHECK-LABEL: @crossLifetimeClonePreserved_readOnly
util.func public @crossLifetimeClonePreserved_readOnly(%input: !stream.resource<external>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK: stream.async.clone
  %clone = stream.async.clone %input : !stream.resource<external>{%size} -> !stream.resource<*>{%size}
  %d = stream.async.dispatch @dispatch_ex::@dispatch(%clone[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  util.return %d : !stream.resource<*>
}

// -----

// Tests that a cross-lifetime clone (external → *) is preserved when the
// clone result is mutated via a tied dispatch (in-place write).

stream.executable private @dispatch_ex {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%in: !stream.binding, %out: !stream.binding) {
      return
    }
  }
}

// CHECK-LABEL: @crossLifetimeClonePreserved_resultMutated
util.func public @crossLifetimeClonePreserved_resultMutated(%input: !stream.resource<external>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[CLONE:.+]] = stream.async.clone
  %clone = stream.async.clone %input : !stream.resource<external>{%size} -> !stream.resource<*>{%size}
  // Tied dispatch mutates clone in-place.
  // CHECK: stream.async.dispatch @dispatch_ex::@dispatch(%[[CLONE]]{{.*}}) : ({{.*}}) -> %[[CLONE]]
  %d = stream.async.dispatch @dispatch_ex::@dispatch(%clone[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> %clone{%size}
  util.return %d : !stream.resource<*>
}

// -----

// Tests that multiple cross-lifetime clones (external → *) are preserved
// by the first pass. The full pipeline (RefineUsage + second ElideAsyncCopies)
// handles their elimination.

stream.executable private @dispatch_ex {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%in: !stream.binding, %out: !stream.binding) {
      return
    }
  }
}

// CHECK-LABEL: @crossLifetimeClonePreserved_multipleReadOnly
util.func public @crossLifetimeClonePreserved_multipleReadOnly(%input: !stream.resource<external>, %size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  // CHECK: stream.async.clone
  %clone0 = stream.async.clone %input : !stream.resource<external>{%size} -> !stream.resource<*>{%size}
  %d0 = stream.async.dispatch @dispatch_ex::@dispatch(%clone0[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // CHECK: stream.async.clone
  %clone1 = stream.async.clone %input : !stream.resource<external>{%size} -> !stream.resource<*>{%size}
  %d1 = stream.async.dispatch @dispatch_ex::@dispatch(%clone1[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // CHECK: util.return
  util.return %d0, %d1 : !stream.resource<*>, !stream.resource<*>
}
