// RUN: iree-opt --split-input-file --iree-stream-elide-async-copies %s | FileCheck %s

// Test case 1: By-value argument (last use) - clone can be elided
// The caller passes the last use, so the callee can consume it directly.

// CHECK-LABEL: @byval_callee
// CHECK-SAME: (%[[ARG:.+]]: !stream.resource<external>)
util.func private @byval_callee(%arg: !stream.resource<external>) {
  %c100 = arith.constant 100 : index
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %arg : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // ... use clone
  util.return
}

// CHECK-LABEL: @byval_caller
util.func public @byval_caller() {
  %c0_i32 = arith.constant 0 : i32
  %c100 = arith.constant 100 : index
  %resource = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c100}
  // This is the last use of %resource - by-value semantics.
  // CHECK: util.call @byval_callee(%[[RESOURCE:.+]])
  util.call @byval_callee(%resource) : (!stream.resource<external>) -> ()
  // %resource is not used after this point.
  util.return
}

// -----

// Test case 2: By-ref argument (not last use) - clone must be preserved
// The caller has additional uses, so the callee must not mutate the original.

// CHECK-LABEL: @byref_callee
// CHECK-SAME: (%[[ARG:.+]]: !stream.resource<external>)
util.func private @byref_callee(%arg: !stream.resource<external>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[ARG]]
  %clone = stream.async.clone %arg : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // Mutate the clone (tied operation).
  // CHECK: stream.async.fill %{{.+}}, %[[CLONE]]
  %filled = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %clone as !stream.resource<external>{%c100}
  util.return
}

// CHECK-LABEL: @byref_caller
util.func public @byref_caller() {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %resource = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c100}
  // Call with by-ref semantics - %resource is used again after call.
  util.call @byref_callee(%resource) : (!stream.resource<external>) -> ()
  // Additional use - this makes the call argument by-ref.
  %dispatch = stream.async.dispatch @ex::@dispatch(%resource[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  util.return
}

// -----

// Test case 3: Mixed semantics - same callee called with by-val and by-ref
// The callee must preserve defensive copies since it can't assume semantics.

// CHECK-LABEL: @mixed_callee
// CHECK-SAME: (%[[ARG:.+]]: !stream.resource<external>)
util.func private @mixed_callee(%arg: !stream.resource<external>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[ARG]]
  %clone = stream.async.clone %arg : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: stream.async.fill %{{.+}}, %[[CLONE]]
  %filled = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %clone as !stream.resource<external>{%c100}
  util.return
}

// CHECK-LABEL: @mixed_caller_byval
util.func public @mixed_caller_byval() {
  %c0_i32 = arith.constant 0 : i32
  %c100 = arith.constant 100 : index
  %resource = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c100}
  // Last use - by-value semantics.
  util.call @mixed_callee(%resource) : (!stream.resource<external>) -> ()
  util.return
}

// CHECK-LABEL: @mixed_caller_byref
util.func public @mixed_caller_byref() {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c100 = arith.constant 100 : index
  %resource = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c100}
  // Not last use - by-ref semantics.
  util.call @mixed_callee(%resource) : (!stream.resource<external>) -> ()
  // Additional use after call.
  %dispatch = stream.async.dispatch @ex::@dispatch(%resource[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  util.return
}

// -----

// Test case 4: Same-type transfer from by-ref arg - always safe to elide
// external->external transfer is a no-op regardless of semantics.

// CHECK-LABEL: @same_type_transfer_byref
// CHECK-SAME: (%[[ARG:.+]]: !stream.resource<external>)
util.func public @same_type_transfer_byref(%arg: !stream.resource<external>) -> !stream.resource<external> {
  %c100 = arith.constant 100 : index
  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %arg : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: util.return %[[ARG]]
  util.return %transfer : !stream.resource<external>
}

// -----

// Test case 5: Type-changing transfer from by-ref arg - must preserve
// transient->external changes mutability semantics.

// CHECK-LABEL: @type_changing_transfer_byref
// CHECK-SAME: (%[[ARG:.+]]: !stream.resource<transient>)
util.func public @type_changing_transfer_byref(%arg: !stream.resource<transient>) -> !stream.resource<external> {
  %c100 = arith.constant 100 : index
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[ARG]]
  %transfer = stream.async.transfer %arg : !stream.resource<transient>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<external>
}

// -----

// Test case 6: Transfer chain with by-ref source - both preserved
// Since the source is a by-ref argument and transfers change types,
// both transfers must be preserved. We can't collapse because the
// intermediate transient type may be semantically important.

// CHECK-LABEL: @transfer_chain_byref
// CHECK-SAME: (%[[ARG:.+]]: !stream.resource<constant>)
util.func public @transfer_chain_byref(%arg: !stream.resource<constant>) -> !stream.resource<external> {
  %c100 = arith.constant 100 : index
  // CHECK: %[[T1:.+]] = stream.async.transfer %[[ARG]] : !stream.resource<constant>{{.+}} -> !stream.resource<transient>
  %t1 = stream.async.transfer %arg : !stream.resource<constant>{%c100} -> !stream.resource<transient>{%c100}
  // CHECK: %[[T2:.+]] = stream.async.transfer %[[T1]] : !stream.resource<transient>{{.+}} -> !stream.resource<external>
  %t2 = stream.async.transfer %t1 : !stream.resource<transient>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: util.return %[[T2]]
  util.return %t2 : !stream.resource<external>
}

// -----

// Test case 7: Clone from by-val arg feeding tied op - can elide

// CHECK-LABEL: @clone_byval_to_tied
util.func public @clone_byval_to_tied() {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  %resource = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c100}
  // Last use - by-value semantics.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %resource : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: stream.async.fill %{{.+}}, %[[RESOURCE:.+]]
  %filled = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %clone as !stream.resource<external>{%c100}
  util.return
}

// -----

// Test case 8: Clone from by-ref arg feeding tied op - must preserve

// CHECK-LABEL: @clone_byref_to_tied
// CHECK-SAME: (%[[ARG:.+]]: !stream.resource<external>)
util.func public @clone_byref_to_tied(%arg: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[ARG]]
  %clone = stream.async.clone %arg : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // Tied op will mutate - must not mutate the original arg.
  // CHECK: %[[FILLED:.+]] = stream.async.fill %{{.+}}, %[[CLONE]]
  %filled = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %clone as !stream.resource<external>{%c100}
  // Return the mutated value, original arg is also returned elsewhere.
  // CHECK: util.return %[[FILLED]]
  util.return %filled : !stream.resource<external>
}

// CHECK-LABEL: @clone_byref_to_tied_caller
util.func public @clone_byref_to_tied_caller() {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c100 = arith.constant 100 : index
  %resource = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c100}
  // Not last use - by-ref.
  %result = util.call @clone_byref_to_tied(%resource) : (!stream.resource<external>) -> !stream.resource<external>
  // Original resource used again.
  %dispatch = stream.async.dispatch @ex::@dispatch(%resource[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  util.return
}

// -----

// Test case 9: Multiple users (fork) - clone must be preserved

// CHECK-LABEL: @fork_semantics
util.func public @fork_semantics() {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c100 = arith.constant 100 : index
  // CHECK: %[[RESOURCE:.+]] = stream.async.splat
  %resource = stream.async.splat %c0_i32 : i32 -> !stream.resource<external>{%c100}
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[RESOURCE]]
  %clone = stream.async.clone %resource : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // Both original and clone are used - fork semantics.
  // CHECK: stream.async.dispatch @ex::@dispatch1(%[[RESOURCE]]
  %dispatch1 = stream.async.dispatch @ex::@dispatch1(%resource[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  // CHECK: stream.async.dispatch @ex::@dispatch2(%[[CLONE]]
  %dispatch2 = stream.async.dispatch @ex::@dispatch2(%clone[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  util.return
}

// -----

// Test case 10: Transfer from external arg to external (same type) with multiple uses
// Same-type transfer is a no-op and can be elided even with multiple uses of source.

// CHECK-LABEL: @same_type_transfer_multi_use
// CHECK-SAME: (%[[ARG:.+]]: !stream.resource<external>)
util.func public @same_type_transfer_multi_use(%arg: !stream.resource<external>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %arg : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // Both original and transfer result used.
  // CHECK: stream.async.dispatch @ex::@dispatch1(%[[ARG]]
  %dispatch1 = stream.async.dispatch @ex::@dispatch1(%arg[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  // CHECK: stream.async.dispatch @ex::@dispatch2(%[[ARG]]
  %dispatch2 = stream.async.dispatch @ex::@dispatch2(%transfer[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  util.return
}
