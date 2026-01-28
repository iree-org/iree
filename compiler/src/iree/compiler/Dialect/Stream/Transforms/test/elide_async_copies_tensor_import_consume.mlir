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

// Tests stream.tensor.import consume with multiple uses (fork).
// Clone must be preserved because original is used again.

// CHECK-LABEL: @consume_import_with_fork
// CHECK-SAME: (%[[BUFFER:[^:]+]]: !util.buffer)
util.func public @consume_import_with_fork(%arg0: !util.buffer) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  // CHECK: %[[IMPORT:.+]] = stream.tensor.import consume %[[BUFFER]]
  %import = stream.tensor.import consume %arg0 : !util.buffer -> tensor<f32> in !stream.resource<external>{%c100}
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[IMPORT]]
  %clone = stream.async.clone %import : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}
  // Both original and clone are used - fork semantics.
  // CHECK: stream.async.dispatch @ex::@dispatch1(%[[IMPORT]][%c0 to %c100 for %c100])
  %dispatch1 = stream.async.dispatch @ex::@dispatch1(%import[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
  // CHECK: stream.async.dispatch @ex::@dispatch2(%[[CLONE]][%c0 to %c100 for %c100])
  %dispatch2 = stream.async.dispatch @ex::@dispatch2(%clone[%c0 to %c100 for %c100]) : (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}
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
