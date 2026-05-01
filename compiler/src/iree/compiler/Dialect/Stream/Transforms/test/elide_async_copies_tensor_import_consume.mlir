// RUN: iree-opt --split-input-file --iree-stream-elide-async-copies %s | FileCheck %s

// Tests that tensor.import with consume attribute enables clone elision.
// The consume attribute indicates transfer of ownership (by-value semantics).

// CHECK-LABEL: @tensor_import_consume_allows_clone_elision
// CHECK-SAME: (%[[BUFFER:[^:]+]]: !util.buffer)
util.func public @tensor_import_consume_allows_clone_elision(%arg0: !util.buffer) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import consume %[[BUFFER]] : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  %import = stream.tensor.import consume %arg0 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %import : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[IMPORT]]
  %fill = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %clone as !stream.resource<external>{%c100}
  // CHECK: util.return %[[FILL]]
  util.return %fill : !stream.resource<external>
}

// -----

// Tests that tensor.import without consume attribute preserves clone when
// the import has multiple uses (by-reference semantics).

// CHECK-LABEL: @tensor_import_no_consume_preserves_clone
// CHECK-SAME: (%[[BUFFER:[^:]+]]: !util.buffer)
util.func public @tensor_import_no_consume_preserves_clone(%arg0: !util.buffer) -> (!stream.resource<external>, !stream.resource<external>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import %[[BUFFER]] : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  %import = stream.tensor.import %arg0 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[IMPORT]]
  %clone = stream.async.clone %import : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[CLONE]]
  %fill = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %clone as !stream.resource<external>{%c100}
  // Import used again - multiple uses means by-reference.
  // CHECK: util.return %[[FILL]], %[[IMPORT]]
  util.return %fill, %import : !stream.resource<external>, !stream.resource<external>
}

// -----

// Tests stream.tensor.import consume through function call.
// The consume attribute enables by-value semantics for the callee.

// CHECK-LABEL: util.func private @callee_with_consume_import
// CHECK-SAME: (%[[ARG:.+]]: !stream.resource<external>)
util.func private @callee_with_consume_import(%arg: !stream.resource<external>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %arg : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: stream.async.fill %c123_i32, %[[ARG]]
  %fill = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %clone as !stream.resource<external>{%c100}
  util.return
}

// CHECK-LABEL: @caller_with_consume_import
// CHECK-SAME: (%[[BUFFER:[^:]+]]: !util.buffer)
util.func public @caller_with_consume_import(%arg0: !util.buffer) {
  %c100 = arith.constant 100 : index
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import consume %[[BUFFER]]
  %import = stream.tensor.import consume %arg0 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // Last use - by-value semantics from consume attribute.
  // CHECK: util.call @callee_with_consume_import(%[[IMPORT]])
  util.call @callee_with_consume_import(%import) : (!stream.resource<external>) -> ()
  util.return
}

// -----

// Tests stream.tensor.import consume with multiple read-only uses.
// Clone can be elided because neither the source nor clone is mutated.
// Both dispatches only read, so sharing the buffer is safe.

// CHECK-LABEL: @consume_import_read_only_fork
// CHECK-SAME: (%[[BUFFER:[^:]+]]: !util.buffer)
util.func public @consume_import_read_only_fork(%arg0: !util.buffer) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import consume %[[BUFFER]]
  %import = stream.tensor.import consume %arg0 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // Clone can be elided - neither original nor clone is mutated.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %import : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // Both dispatches read from the same buffer (safe - no mutation).
  // CHECK: stream.async.dispatch @ex::@dispatch1(%[[IMPORT]][%c0 to %c100 for %c100])
  %dispatch1 = stream.async.dispatch @ex::@dispatch1(%import[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  // CHECK: stream.async.dispatch @ex::@dispatch2(%[[IMPORT]][%c0 to %c100 for %c100])
  %dispatch2 = stream.async.dispatch @ex::@dispatch2(%clone[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  util.return
}

// -----

// Tests stream.tensor.import consume with fork where mutation occurs.
// Clone must be preserved because the clone result is mutated via tied dispatch.

// CHECK-LABEL: @consume_import_with_mutating_fork
// CHECK-SAME: (%[[BUFFER:[^:]+]]: !util.buffer)
util.func public @consume_import_with_mutating_fork(%arg0: !util.buffer) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import consume %[[BUFFER]]
  %import = stream.tensor.import consume %arg0 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // Clone must be preserved - clone result is mutated by tied dispatch.
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[IMPORT]]
  %clone = stream.async.clone %import : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // First dispatch only reads from import.
  // CHECK: stream.async.dispatch @ex::@dispatch1(%[[IMPORT]][%c0 to %c100 for %c100])
  %dispatch1 = stream.async.dispatch @ex::@dispatch1(%import[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  // Second dispatch mutates the clone in-place (tied operand).
  // CHECK: stream.async.dispatch @ex::@dispatch2(%[[CLONE]][%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> %[[CLONE]]{%c100}
  %dispatch2 = stream.async.dispatch @ex::@dispatch2(%clone[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> %clone{%c100}
  util.return
}

// -----

// Tests stream.tensor.import consume feeding transfer.
// Same-type transfer from consumed import can be elided.

// CHECK-LABEL: @consume_import_to_transfer
// CHECK-SAME: (%[[BUFFER:[^:]+]]: !util.buffer)
util.func public @consume_import_to_transfer(%arg0: !util.buffer) -> !stream.resource<external> {
  %c100 = arith.constant 100 : index
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import consume %[[BUFFER]]
  %import = stream.tensor.import consume %arg0 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %import : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: util.return %[[IMPORT]]
  util.return %transfer : !stream.resource<external>
}

// -----

// Tests stream.tensor.import consume with type-changing transfer.
// Type-changing transfer from consumed import should be preserved.

// CHECK-LABEL: @consume_import_type_changing_transfer
// CHECK-SAME: (%[[BUFFER:[^:]+]]: !util.buffer)
util.func public @consume_import_type_changing_transfer(%arg0: !util.buffer) -> !stream.resource<transient> {
  %c100 = arith.constant 100 : index
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import consume %[[BUFFER]] : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  %import = stream.tensor.import consume %arg0 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[IMPORT]] : !stream.resource<external>{%c100} -> !stream.resource<transient>{%c100}
  %transfer = stream.async.transfer %import : !stream.resource<external>{%c100} -> !stream.resource<transient>{%c100}
  // CHECK: util.return %[[TRANSFER]]
  util.return %transfer : !stream.resource<transient>
}

// -----

// Tests stream.tensor.import consume in scf.if.
// Consume attribute enables elision in conditional branches when last use.

// CHECK-LABEL: @consume_import_in_if
// CHECK-SAME: (%{{[^:]+}}: i1, %[[BUFFER:[^:]+]]: !util.buffer)
util.func public @consume_import_in_if(%arg0: i1, %arg1: !util.buffer) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import consume %[[BUFFER]]
  %import = stream.tensor.import consume %arg1 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // CHECK: %[[RESULT:.+]] = scf.if {{.+}}
  %result = scf.if %arg0 -> (!stream.resource<external>) {
    // CHECK-NOT: stream.async.clone
    %clone = stream.async.clone %import : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
    // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[IMPORT]]
    %fill = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %clone as !stream.resource<external>{%c100}
    // CHECK: scf.yield %[[FILL]]
    scf.yield %fill : !stream.resource<external>
  } else {
    // CHECK: scf.yield %[[IMPORT]]
    scf.yield %import : !stream.resource<external>
  }
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<external>
}

// -----

// Tests stream.tensor.import consume passed to function with additional use.
// The callee receives by-ref argument (not last use) but can still use tied ops.

// CHECK-LABEL: util.func private @callee_with_tied_op
// CHECK-SAME: (%[[ARG:.+]]: !stream.resource<external>)
util.func private @callee_with_tied_op(%arg: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // Tied operation can mutate in-place without explicit clone.
  // CHECK-NOT: stream.async.clone
  // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[ARG]]
  %fill = stream.async.fill %c123_i32, %arg[%c0 to %c100 for %c100] : i32 -> %arg as !stream.resource<external>{%c100}
  // CHECK: util.return %[[FILL]]
  util.return %fill : !stream.resource<external>
}

// CHECK-LABEL: @caller_with_post_call_use
// CHECK-SAME: (%[[BUFFER:[^:]+]]: !util.buffer)
util.func public @caller_with_post_call_use(%arg0: !util.buffer) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import consume %[[BUFFER]]
  %import = stream.tensor.import consume %arg0 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // Not last use - by-ref semantics despite consume attribute.
  %result = util.call @callee_with_tied_op(%import) : (!stream.resource<external>) -> !stream.resource<external>
  // Additional use after call.
  // CHECK: stream.async.dispatch @ex::@dispatch(%[[IMPORT]][%c0 to %c100 for %c100])
  %dispatch = stream.async.dispatch @ex::@dispatch(%import[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  util.return
}

// -----

// Tests stream.tensor.import consume with reuse as output.
// When consumed, the input buffer can be reused for output.

// CHECK-LABEL: @consume_import_reuse_as_output
// CHECK-SAME: (%[[BUFFER:[^:]+]]: !util.buffer)
util.func public @consume_import_reuse_as_output(%arg0: !util.buffer) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import consume %[[BUFFER]]
  %import = stream.tensor.import consume %arg0 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // CHECK-NOT: stream.async.clone
  // CHECK: %[[DISPATCH:.+]] = stream.async.dispatch @ex::@dispatch(%[[IMPORT]][%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> %[[IMPORT]]{%c100}
  %dispatch = stream.async.dispatch @ex::@dispatch(%import[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> %import{%c100}
  // CHECK: util.return %[[DISPATCH]]
  util.return %dispatch : !stream.resource<external>
}

// -----

// Tests mixed stream.tensor.import (with and without consume).
// Clone preservation depends on consume attribute and usage.

// CHECK-LABEL: @mixed_consume_no_consume
// CHECK-SAME: (%[[BUFFER0:[^:]+]]: !util.buffer, %[[BUFFER1:[^:]+]]: !util.buffer)
util.func public @mixed_consume_no_consume(%arg0: !util.buffer, %arg1: !util.buffer) -> (!stream.resource<external>, !stream.resource<external>, !stream.resource<external>) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32

  // With consume - clone can be elided (last use).
  // CHECK: %[[IMPORT0:.+]] = stream.tensor.import consume %[[BUFFER0]]
  %import0 = stream.tensor.import consume %arg0 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // CHECK-NOT: stream.async.clone %[[IMPORT0]]
  %clone0 = stream.async.clone %import0 : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: %[[FILL0:.+]] = stream.async.fill %c123_i32, %[[IMPORT0]]
  %fill0 = stream.async.fill %c123_i32, %clone0[%c0 to %c100 for %c100] : i32 -> %clone0 as !stream.resource<external>{%c100}

  // Without consume - clone must be preserved (multiple uses).
  // CHECK: %[[IMPORT1:.+]] = stream.tensor.import %[[BUFFER1]]
  %import1 = stream.tensor.import %arg1 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // CHECK: %[[CLONE1:.+]] = stream.async.clone %[[IMPORT1]]
  %clone1 = stream.async.clone %import1 : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: %[[FILL1:.+]] = stream.async.fill %c123_i32, %[[CLONE1]]
  %fill1 = stream.async.fill %c123_i32, %clone1[%c0 to %c100 for %c100] : i32 -> %clone1 as !stream.resource<external>{%c100}

  // CHECK: util.return %[[FILL0]], %[[FILL1]], %[[IMPORT1]]
  util.return %fill0, %fill1, %import1 : !stream.resource<external>, !stream.resource<external>, !stream.resource<external>
}

// -----

// Tests that a non-consuming import remains borrowed even when the imported SSA
// value has no uses after the clone. The clone protects caller-owned storage
// from the tied fill mutation and must not be elided by one-use reasoning.

// CHECK-LABEL: @borrowed_import_single_use_preserves_mutating_clone
// CHECK-SAME: (%[[BUFFER:[^:]+]]: !util.buffer)
util.func public @borrowed_import_single_use_preserves_mutating_clone(%arg0: !util.buffer) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import %[[BUFFER]]
  %import = stream.tensor.import %arg0 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[IMPORT]]
  %clone = stream.async.clone %import : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[CLONE]]
  %fill = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %clone as !stream.resource<external>{%c100}
  // CHECK: util.return %[[FILL]]
  util.return %fill : !stream.resource<external>
}

// -----

// Tests that borrowed storage is preserved across call boundaries. Even though
// the imported SSA value is a last-use call operand, the callee argument still
// aliases non-consuming imported storage and needs its protective clone.

// CHECK-LABEL: util.func private @borrowed_import_last_use_callee
// CHECK-SAME: (%[[ARG:.+]]: !stream.resource<external>)
util.func private @borrowed_import_last_use_callee(%arg: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[ARG]]
  %clone = stream.async.clone %arg : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[CLONE]]
  %fill = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %clone as !stream.resource<external>{%c100}
  // CHECK: util.return %[[FILL]]
  util.return %fill : !stream.resource<external>
}

// CHECK-LABEL: @borrowed_import_last_use_caller
// CHECK-SAME: (%[[BUFFER:[^:]+]]: !util.buffer)
util.func public @borrowed_import_last_use_caller(%arg0: !util.buffer) -> !stream.resource<external> {
  %c100 = arith.constant 100 : index
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import %[[BUFFER]]
  %import = stream.tensor.import %arg0 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // CHECK: %[[RESULT:.+]] = util.call @borrowed_import_last_use_callee(%[[IMPORT]])
  %result = util.call @borrowed_import_last_use_callee(%import) : (!stream.resource<external>) -> !stream.resource<external>
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<external>
}

// -----

// Tests that borrowed imports still permit clone elision when the clone result
// is read-only. Non-consuming imports only require protective clones when the
// clone would otherwise shield caller-owned storage from mutation.

stream.executable private @ex_borrowed_readonly {
  stream.executable.export public @read_only workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @borrowed_import_readonly_elides_clone
// CHECK-SAME: (%[[BUFFER:[^:]+]]: !util.buffer)
util.func public @borrowed_import_readonly_elides_clone(%arg0: !util.buffer) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import %[[BUFFER]]
  %import = stream.tensor.import %arg0 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %import : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex_borrowed_readonly::@read_only(%[[IMPORT]]
  %result = stream.async.dispatch @ex_borrowed_readonly::@read_only(%clone[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<external>
}

// -----

// Tests that borrowed storage remains borrowed through a same-lifetime transfer
// that topology analysis cannot remove. The transfer may move the storage
// between devices, but it does not create an owned copy with a new lifetime.

module @test attributes {
  stream.topology = #hal.device.topology<links = [
    (@dev_a -> @dev_c = {})
  ]>
} {

// CHECK-LABEL: @borrowed_import_transfer_preserves_mutating_clone
// CHECK-SAME: (%[[BUFFER:[^:]+]]: !util.buffer)
util.func public @borrowed_import_transfer_preserves_mutating_clone(%arg0: !util.buffer) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import on(#hal.device.promise<@dev_a>) %[[BUFFER]]
  %import = stream.tensor.import on(#hal.device.promise<@dev_a>) %arg0 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // CHECK: %[[TRANSFER:.+]] = stream.async.transfer %[[IMPORT]]
  // CHECK-SAME: from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_c>)
  %transfer = stream.async.transfer %import : !stream.resource<external>{%c100}
      from(#hal.device.promise<@dev_a>) -> to(#hal.device.promise<@dev_c>) !stream.resource<external>{%c100}
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[TRANSFER]]
  %clone = stream.async.clone %transfer : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[CLONE]]
  %fill = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %clone as !stream.resource<external>{%c100}
  // CHECK: util.return %[[FILL]]
  util.return %fill : !stream.resource<external>
}

} // module
