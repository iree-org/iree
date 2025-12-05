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

// -----

//===----------------------------------------------------------------------===//
// Borrowed buffer tests (issue #16168)
//===----------------------------------------------------------------------===//

// Tests that a clone from a non-consuming import (borrowed buffer) must be
// preserved when the clone result is mutated. This prevents in-place mutation
// of caller-owned buffers.
//
// This is the matmul_accumulate pattern:
//   1. Import accumulator buffer (caller retains ownership)
//   2. Clone to protect from in-place mutation
//   3. Dispatch writes result into clone (tied operand)
//   4. Export result
//
// Without the protective clone, the dispatch would mutate the original buffer,
// corrupting the caller's data.

stream.executable private @ex_matmul {
  stream.executable.export public @matmul_accumulate workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @borrowed_import_mutated_must_preserve_clone
// CHECK-SAME: (%[[LHS_VIEW:.+]]: !util.buffer, %[[RHS_VIEW:.+]]: !util.buffer, %[[ACC_VIEW:.+]]: !util.buffer)
util.func public @borrowed_import_mutated_must_preserve_clone(
    %lhs_view: !util.buffer, %rhs_view: !util.buffer, %acc_view: !util.buffer) -> !util.buffer {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index

  // Import LHS (read-only, no clone needed).
  // CHECK: %[[LHS:.+]] = stream.tensor.import %[[LHS_VIEW]] :
  %lhs = stream.tensor.import %lhs_view : !util.buffer -> tensor<1x1xi8> in !stream.resource<external>{%c4}

  // Import RHS (read-only, no clone needed).
  // CHECK: %[[RHS:.+]] = stream.tensor.import %[[RHS_VIEW]] :
  %rhs = stream.tensor.import %rhs_view : !util.buffer -> tensor<1x1xi8> in !stream.resource<external>{%c4}

  // Import accumulator WITHOUT consume attribute - caller retains ownership.
  // This is the "borrowed" case where we must not mutate the original.
  // CHECK: %[[ACC:.+]] = stream.tensor.import %[[ACC_VIEW]] :
  %acc = stream.tensor.import %acc_view : !util.buffer -> tensor<1x1xi32> in !stream.resource<external>{%c4}

  // Clone MUST be preserved because:
  //   1. Source is a non-consuming import (borrowed buffer)
  //   2. Clone result is mutated (tied to dispatch output)
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[ACC]]
  %clone = stream.async.clone %acc : !stream.resource<external>{%c4} -> !stream.resource<external>{%c4}

  // Dispatch writes result into clone (tied operand).
  // CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex_matmul::@matmul_accumulate(%[[LHS]]{{.+}}, %[[RHS]]{{.+}}, %[[CLONE]]{{.+}}) : {{.+}} -> %[[CLONE]]
  %result = stream.async.dispatch @ex_matmul::@matmul_accumulate(
      %lhs[%c0 to %c4 for %c4], %rhs[%c0 to %c4 for %c4], %clone[%c0 to %c4 for %c4]) :
      (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}, !stream.resource<external>{%c4}) -> %clone{%c4}

  // Export result.
  // CHECK: %[[EXPORTED:.+]] = stream.tensor.export %[[RESULT]]
  %exported = stream.tensor.export %result : tensor<1x1xi32> in !stream.resource<external>{%c4} -> !util.buffer

  // CHECK: util.return %[[EXPORTED]]
  util.return %exported : !util.buffer
}

// -----

// Tests that a clone from a consuming import CAN be elided when mutated.
// The `consume` attribute indicates caller transfers ownership, so in-place
// mutation is safe.

stream.executable private @ex_matmul_consume {
  stream.executable.export public @matmul_accumulate workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @consuming_import_mutated_can_elide_clone
// CHECK-SAME: (%[[LHS_VIEW:.+]]: !util.buffer, %[[RHS_VIEW:.+]]: !util.buffer, %[[ACC_VIEW:.+]]: !util.buffer)
util.func public @consuming_import_mutated_can_elide_clone(
    %lhs_view: !util.buffer, %rhs_view: !util.buffer, %acc_view: !util.buffer) -> !util.buffer {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index

  // Import LHS.
  // CHECK: %[[LHS:.+]] = stream.tensor.import %[[LHS_VIEW]] :
  %lhs = stream.tensor.import %lhs_view : !util.buffer -> tensor<1x1xi8> in !stream.resource<external>{%c4}

  // Import RHS.
  // CHECK: %[[RHS:.+]] = stream.tensor.import %[[RHS_VIEW]] :
  %rhs = stream.tensor.import %rhs_view : !util.buffer -> tensor<1x1xi8> in !stream.resource<external>{%c4}

  // Import accumulator WITH consume attribute - caller transfers ownership.
  // CHECK: %[[ACC:.+]] = stream.tensor.import consume %[[ACC_VIEW]] :
  %acc = stream.tensor.import consume %acc_view : !util.buffer -> tensor<1x1xi32> in !stream.resource<external>{%c4}

  // Clone CAN be elided because:
  //   1. Source is a consuming import (ownership transferred)
  //   2. We own the buffer and can mutate in-place
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %acc : !stream.resource<external>{%c4} -> !stream.resource<external>{%c4}

  // Dispatch writes directly into imported buffer.
  // CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex_matmul_consume::@matmul_accumulate(%[[LHS]]{{.+}}, %[[RHS]]{{.+}}, %[[ACC]]{{.+}}) : {{.+}} -> %[[ACC]]
  %result = stream.async.dispatch @ex_matmul_consume::@matmul_accumulate(
      %lhs[%c0 to %c4 for %c4], %rhs[%c0 to %c4 for %c4], %clone[%c0 to %c4 for %c4]) :
      (!stream.resource<external>{%c4}, !stream.resource<external>{%c4}, !stream.resource<external>{%c4}) -> %clone{%c4}

  // Export result.
  // CHECK: %[[EXPORTED:.+]] = stream.tensor.export %[[RESULT]]
  %exported = stream.tensor.export %result : tensor<1x1xi32> in !stream.resource<external>{%c4} -> !util.buffer

  // CHECK: util.return %[[EXPORTED]]
  util.return %exported : !util.buffer
}

// -----

// Tests borrowed import passed through a function call.
// Even when the import is in the caller and passed to callee, the borrowed
// semantics must be preserved across the call boundary.

stream.executable private @ex_callee {
  stream.executable.export public @mutate workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: util.func private @borrowed_through_call_callee
// CHECK-SAME: (%[[ARG:.+]]: !stream.resource<external>, %[[SIZE:.+]]: index)
util.func private @borrowed_through_call_callee(%arg: !stream.resource<external>, %size: index) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  // The callee receives a by-ref argument (not last use in caller).
  // Clone must be preserved because we don't know if caller owns the buffer.
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[ARG]]
  %clone = stream.async.clone %arg : !stream.resource<external>{%size} -> !stream.resource<external>{%size}
  // CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex_callee::@mutate(%[[CLONE]]{{.+}}) : {{.+}} -> %[[CLONE]]
  %result = stream.async.dispatch @ex_callee::@mutate(%clone[%c0 to %size for %size]) : (!stream.resource<external>{%size}) -> %clone{%size}
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<external>
}

// CHECK-LABEL: @borrowed_through_call_caller
// CHECK-SAME: (%[[VIEW:.+]]: !util.buffer)
util.func public @borrowed_through_call_caller(%view: !util.buffer) -> (!util.buffer, !util.buffer) {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index

  // Import without consume - borrowed.
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import %[[VIEW]] :
  %imported = stream.tensor.import %view : !util.buffer -> tensor<100xf32> in !stream.resource<external>{%c100}

  // Call function with imported buffer (not last use).
  %result = util.call @borrowed_through_call_callee(%imported, %c100) : (!stream.resource<external>, index) -> !stream.resource<external>

  // Original imported buffer used again - proves it wasn't mutated.
  // CHECK: %[[EXPORT1:.+]] = stream.tensor.export
  %export1 = stream.tensor.export %imported : tensor<100xf32> in !stream.resource<external>{%c100} -> !util.buffer

  // Result from callee.
  // CHECK: %[[EXPORT2:.+]] = stream.tensor.export
  %export2 = stream.tensor.export %result : tensor<100xf32> in !stream.resource<external>{%c100} -> !util.buffer

  // CHECK: util.return %[[EXPORT1]], %[[EXPORT2]]
  util.return %export1, %export2 : !util.buffer, !util.buffer
}

// -----

// Tests that borrowed import with read-only use CAN have clone elided.
// If the clone result is never mutated, the clone is unnecessary.

stream.executable private @ex_readonly {
  stream.executable.export public @read_only workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @borrowed_import_readonly_can_elide_clone
// CHECK-SAME: (%[[VIEW:.+]]: !util.buffer)
util.func public @borrowed_import_readonly_can_elide_clone(%view: !util.buffer) -> !util.buffer {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index

  // Import without consume - borrowed.
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import %[[VIEW]] :
  %imported = stream.tensor.import %view : !util.buffer -> tensor<100xf32> in !stream.resource<external>{%c100}

  // Clone can be elided because dispatch only reads (no tied output).
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %imported : !stream.resource<external>{%c100} -> !stream.resource<external>{%c100}

  // Dispatch reads from clone but produces NEW output (not tied).
  // CHECK: %[[RESULT:.+]] = stream.async.dispatch @ex_readonly::@read_only(%[[IMPORTED]]{{.+}}) : {{.+}} -> !stream.resource<external>
  %result = stream.async.dispatch @ex_readonly::@read_only(%clone[%c0 to %c100 for %c100]) :
      (!stream.resource<external>{%c100}) -> !stream.resource<external>{%c100}

  // CHECK: %[[EXPORTED:.+]] = stream.tensor.export %[[RESULT]]
  %exported = stream.tensor.export %result : tensor<100xf32> in !stream.resource<external>{%c100} -> !util.buffer

  // CHECK: util.return %[[EXPORTED]]
  util.return %exported : !util.buffer
}

// -----

//===----------------------------------------------------------------------===//
// Full overwrite tests (stream.async.update)
//===----------------------------------------------------------------------===//

// Tests that when stream.async.update fully overwrites a borrowed buffer,
// the data is no longer considered borrowed - clones can be elided.

// CHECK-LABEL: @update_full_overwrite_elides_clone
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer)
util.func public @update_full_overwrite_elides_clone(%arg0: !util.buffer) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c123_i32 = arith.constant 123 : i32

  // Import without consume - normally "borrowed".
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import %[[BUFFER]] :
  %imported = stream.tensor.import %arg0 : !util.buffer -> tensor<64xf32> in !stream.resource<external>{%c256}

  // Allocate new data to write into the buffer (same lifetime as target).
  %alloc, %alloc_tp = stream.resource.alloca uninitialized : !stream.resource<external>{%c256} => !stream.timepoint
  %ready = stream.timepoint.await %alloc_tp => %alloc : !stream.resource<external>{%c256}

  // Full overwrite: [0, 256) of 256-byte buffer.
  // After this, the data in %updated is NEW, not borrowed.
  // CHECK: %[[UPDATED:.+]] = stream.async.update
  %updated = stream.async.update %ready, %imported[%c0 to %c256] :
      !stream.resource<external>{%c256} -> %imported as !stream.resource<external>{%c256}

  // Clone of fully-overwritten buffer CAN be elided.
  // The data is no longer "borrowed" - we just wrote it.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %updated : !stream.resource<external>{%c256} -> !stream.resource<external>{%c256}

  // Mutation to verify elision (fill mutates the clone).
  // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[UPDATED]]
  %fill = stream.async.fill %c123_i32, %clone[%c0 to %c256 for %c256] : i32 -> %clone as !stream.resource<external>{%c256}

  // CHECK: util.return %[[FILL]]
  util.return %fill : !stream.resource<external>
}

// -----

// Tests that partial overwrite still preserves clone.
// Only full overwrites break the "borrowed" chain.

// CHECK-LABEL: @update_partial_overwrite_preserves_clone
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer)
util.func public @update_partial_overwrite_preserves_clone(%arg0: !util.buffer) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c123_i32 = arith.constant 123 : i32

  // Import without consume - borrowed.
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import %[[BUFFER]] :
  %imported = stream.tensor.import %arg0 : !util.buffer -> tensor<64xf32> in !stream.resource<external>{%c256}

  // Allocate new data (smaller than buffer, same lifetime).
  %alloc, %alloc_tp = stream.resource.alloca uninitialized : !stream.resource<external>{%c128} => !stream.timepoint
  %ready = stream.timepoint.await %alloc_tp => %alloc : !stream.resource<external>{%c128}

  // Partial overwrite: [0, 128) of 256-byte buffer.
  // The remaining [128, 256) still contains borrowed data.
  // CHECK: %[[UPDATED:.+]] = stream.async.update
  %updated = stream.async.update %ready, %imported[%c0 to %c128] :
      !stream.resource<external>{%c128} -> %imported as !stream.resource<external>{%c256}

  // Clone must be preserved because buffer is still partially borrowed.
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[UPDATED]]
  %clone = stream.async.clone %updated : !stream.resource<external>{%c256} -> !stream.resource<external>{%c256}

  // Mutation to verify clone is preserved (fill mutates the clone).
  // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[CLONE]]
  %fill = stream.async.fill %c123_i32, %clone[%c0 to %c256 for %c256] : i32 -> %clone as !stream.resource<external>{%c256}

  // CHECK: util.return %[[FILL]]
  util.return %fill : !stream.resource<external>
}

// -----

//===----------------------------------------------------------------------===//
// Lifetime-changing clone elision tests
//===----------------------------------------------------------------------===//

// Tests that external -> * clone can be elided when the result flows through
// ops that can have their types updated (barrier, await, export). This avoids
// unnecessary allocations when external resources are used internally.

// CHECK-LABEL: @external_to_unknown_clone_elision_via_barrier
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer)
util.func public @external_to_unknown_clone_elision_via_barrier(%arg0: !util.buffer) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index

  // Import with consume - we own this buffer.
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import consume %[[BUFFER]] :
  %imported = stream.tensor.import consume %arg0 : !util.buffer -> tensor<64xf32> in !stream.resource<external>{%c256}

  // Clone external -> * would normally be kept because it changes lifetime.
  // But since the result only flows through barrier/await back to external,
  // we can elide it by propagating the external lifetime.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %imported : !stream.resource<external>{%c256} -> !stream.resource<*>{%c256}

  // Barrier that would force * lifetime, but gets updated to external.
  // CHECK: %[[BARRIER:.+]], %{{.+}} = stream.timepoint.barrier{{.+}}%[[IMPORTED]] : !stream.resource<external>
  %barrier, %tp = stream.timepoint.barrier %clone : !stream.resource<*>{%c256} => !stream.timepoint

  // Clone * -> external becomes same-type after propagation and is elided.
  // CHECK-NOT: stream.async.clone
  %final = stream.async.clone %barrier : !stream.resource<*>{%c256} -> !stream.resource<external>{%c256}

  // CHECK: util.return
  util.return %final : !stream.resource<external>
}

// -----

// Tests lifetime propagation through stream.timepoint.await.
// The clone can be elided and the await type updated to external.

// CHECK-LABEL: @external_to_unknown_clone_elision_via_await
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer)
util.func public @external_to_unknown_clone_elision_via_await(%arg0: !util.buffer) -> !stream.resource<external> {
  %c256 = arith.constant 256 : index

  // Import with consume - we own this buffer.
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import consume %[[BUFFER]] :
  %imported = stream.tensor.import consume %arg0 : !util.buffer -> tensor<64xf32> in !stream.resource<external>{%c256}

  // Clone external -> * can be elided because result flows through barrier/await.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %imported : !stream.resource<external>{%c256} -> !stream.resource<*>{%c256}

  // Barrier type propagates from external.
  %barrier, %tp = stream.timepoint.barrier %clone : !stream.resource<*>{%c256} => !stream.timepoint

  // Await type is updated to external through propagation.
  // CHECK: stream.timepoint.await{{.*}}!stream.resource<external>
  %await = stream.timepoint.await %tp => %barrier : !stream.resource<*>{%c256}

  // Clone * -> external becomes same-type after propagation and is elided.
  // CHECK-NOT: stream.async.clone
  %final = stream.async.clone %await : !stream.resource<*>{%c256} -> !stream.resource<external>{%c256}

  util.return %final : !stream.resource<external>
}

// -----

// Tests lifetime propagation through a mixed chain of barrier and await ops.
// All clones should be elided and all types propagated to external.

// CHECK-LABEL: @external_to_unknown_clone_elision_via_mixed_chain
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer)
util.func public @external_to_unknown_clone_elision_via_mixed_chain(%arg0: !util.buffer) -> !stream.resource<external> {
  %c256 = arith.constant 256 : index

  // Import with consume - we own this buffer.
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import consume %[[BUFFER]] :
  %imported = stream.tensor.import consume %arg0 : !util.buffer -> tensor<64xf32> in !stream.resource<external>{%c256}

  // Clone external -> * can be elided through the entire chain.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %imported : !stream.resource<external>{%c256} -> !stream.resource<*>{%c256}

  // First barrier/await pair.
  // CHECK: stream.timepoint.barrier{{.*}}!stream.resource<external>
  %barrier1, %tp1 = stream.timepoint.barrier %clone : !stream.resource<*>{%c256} => !stream.timepoint
  // CHECK: stream.timepoint.await{{.*}}!stream.resource<external>
  %await1 = stream.timepoint.await %tp1 => %barrier1 : !stream.resource<*>{%c256}

  // Second barrier/await pair.
  // CHECK: stream.timepoint.barrier{{.*}}!stream.resource<external>
  %barrier2, %tp2 = stream.timepoint.barrier %await1 : !stream.resource<*>{%c256} => !stream.timepoint
  // CHECK: stream.timepoint.await{{.*}}!stream.resource<external>
  %await2 = stream.timepoint.await %tp2 => %barrier2 : !stream.resource<*>{%c256}

  // Clone * -> external becomes same-type after propagation and is elided.
  // CHECK-NOT: stream.async.clone
  %final = stream.async.clone %await2 : !stream.resource<*>{%c256} -> !stream.resource<external>{%c256}

  util.return %final : !stream.resource<external>
}

// -----

// Tests that lifetime propagation is blocked by dispatch ops that produce
// new resources (not tied to input). The clone must be preserved.

stream.executable private @ex_producer {
  stream.executable.export public @produce workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
}

// CHECK-LABEL: @lifetime_propagation_blocked_by_dispatch
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer)
util.func public @lifetime_propagation_blocked_by_dispatch(%arg0: !util.buffer) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index

  // Import with consume - we own this buffer.
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import consume %[[BUFFER]] :
  %imported = stream.tensor.import consume %arg0 : !util.buffer -> tensor<64xf32> in !stream.resource<external>{%c256}

  // Clone should NOT be elided - dispatch produces new resource, breaks chain.
  // CHECK: stream.async.clone{{.*}}!stream.resource<external>{{.*}}-> !stream.resource<*>
  %clone = stream.async.clone %imported : !stream.resource<external>{%c256} -> !stream.resource<*>{%c256}

  // Dispatch produces NEW resource (not tied to operand) - this breaks the
  // propagation chain since we cannot update dispatch output types.
  %dispatch = stream.async.dispatch @ex_producer::@produce(%clone[%c0 to %c256 for %c256]) :
      (!stream.resource<*>{%c256}) -> !stream.resource<*>{%c256}

  // Final clone * -> external can't be elided for the initial clone.
  %final = stream.async.clone %dispatch : !stream.resource<*>{%c256} -> !stream.resource<external>{%c256}

  util.return %final : !stream.resource<external>
}

// -----

//===----------------------------------------------------------------------===//
// Feature interaction tests (borrowed + lifetime propagation + mutation)
//===----------------------------------------------------------------------===//

// Tests the critical case: borrowed source + lifetime propagation + mutation.
// Clone must be PRESERVED because:
// 1. Source is borrowed (no consume)
// 2. Result is mutated (fill operation)
// Without the clone, we would corrupt the caller's buffer.

// CHECK-LABEL: @borrowed_external_propagate_and_mutate
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer)
util.func public @borrowed_external_propagate_and_mutate(%arg0: !util.buffer) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c123_i32 = arith.constant 123 : i32

  // Import WITHOUT consume - this is a borrowed buffer.
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import %[[BUFFER]] :
  %imported = stream.tensor.import %arg0 : !util.buffer -> tensor<64xf32> in !stream.resource<external>{%c256}

  // Clone MUST be preserved: borrowed source AND result is mutated.
  // CHECK: stream.async.clone{{.*}}!stream.resource<external>{{.*}}-> !stream.resource<*>
  %clone = stream.async.clone %imported : !stream.resource<external>{%c256} -> !stream.resource<*>{%c256}

  // Barrier/await chain - these are pass-throughs, not mutations.
  %barrier, %tp = stream.timepoint.barrier %clone : !stream.resource<*>{%c256} => !stream.timepoint
  %ready = stream.timepoint.await %tp => %barrier : !stream.resource<*>{%c256}

  // Fill MUTATES the resource - this is why clone must be preserved.
  %fill = stream.async.fill %c123_i32, %ready[%c0 to %c256 for %c256] : i32 -> %ready as !stream.resource<*>{%c256}

  %final = stream.async.clone %fill : !stream.resource<*>{%c256} -> !stream.resource<external>{%c256}

  util.return %final : !stream.resource<external>
}

// -----

// Tests borrowed source + lifetime propagation WITHOUT mutation.
// Clone CAN be elided because:
// 1. Even though source is borrowed, result is NOT mutated
// 2. Read-only use of borrowed data is safe
// Combined with lifetime propagation, all ops get updated to external.

// CHECK-LABEL: @borrowed_external_propagate_no_mutate
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer)
util.func public @borrowed_external_propagate_no_mutate(%arg0: !util.buffer) -> !stream.resource<external> {
  %c256 = arith.constant 256 : index

  // Import WITHOUT consume - this is a borrowed buffer.
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import %[[BUFFER]] :
  %imported = stream.tensor.import %arg0 : !util.buffer -> tensor<64xf32> in !stream.resource<external>{%c256}

  // Clone CAN be elided: borrowed but result is NOT mutated.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %imported : !stream.resource<external>{%c256} -> !stream.resource<*>{%c256}

  // Barrier/await chain - pass-throughs, types updated to external.
  // CHECK: stream.timepoint.barrier{{.*}}!stream.resource<external>
  %barrier, %tp = stream.timepoint.barrier %clone : !stream.resource<*>{%c256} => !stream.timepoint
  // CHECK: stream.timepoint.await{{.*}}!stream.resource<external>
  %ready = stream.timepoint.await %tp => %barrier : !stream.resource<*>{%c256}

  // No mutation - just pass through to return.
  // CHECK-NOT: stream.async.clone
  %final = stream.async.clone %ready : !stream.resource<*>{%c256} -> !stream.resource<external>{%c256}

  util.return %final : !stream.resource<external>
}

// -----

// Tests that a full overwrite breaks the borrowed chain, enabling lifetime
// propagation even when the original import was borrowed.
// After a full overwrite, the data is OUR new data, not borrowed.

// CHECK-LABEL: @full_overwrite_breaks_borrowed_then_propagates
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer)
util.func public @full_overwrite_breaks_borrowed_then_propagates(%arg0: !util.buffer) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index

  // Import WITHOUT consume - starts as borrowed.
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import %[[BUFFER]] :
  %imported = stream.tensor.import %arg0 : !util.buffer -> tensor<64xf32> in !stream.resource<external>{%c256}

  // Allocate new data to write.
  %alloc, %alloc_tp = stream.resource.alloca uninitialized : !stream.resource<external>{%c256} => !stream.timepoint
  %ready_alloc = stream.timepoint.await %alloc_tp => %alloc : !stream.resource<external>{%c256}

  // Full overwrite: [0, 256) of 256-byte buffer.
  // After this, %updated contains OUR data, not borrowed data.
  // CHECK: %[[UPDATED:.+]] = stream.async.update
  %updated = stream.async.update %ready_alloc, %imported[%c0 to %c256] :
      !stream.resource<external>{%c256} -> %imported as !stream.resource<external>{%c256}

  // Clone CAN be elided because:
  // 1. The full overwrite broke the borrowed chain
  // 2. %updated is now "owned" data
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %updated : !stream.resource<external>{%c256} -> !stream.resource<*>{%c256}

  // Lifetime propagation works because source is now owned.
  // CHECK: stream.timepoint.barrier{{.*}}!stream.resource<external>
  %barrier, %tp = stream.timepoint.barrier %clone : !stream.resource<*>{%c256} => !stream.timepoint
  %ready = stream.timepoint.await %tp => %barrier : !stream.resource<*>{%c256}

  // CHECK-NOT: stream.async.clone
  %final = stream.async.clone %ready : !stream.resource<*>{%c256} -> !stream.resource<external>{%c256}

  util.return %final : !stream.resource<external>
}

// -----

//===----------------------------------------------------------------------===//
// Mutation analysis tests
//===----------------------------------------------------------------------===//

// Tests that stream.async.transfer BREAKS the aliasing chain.
// Transfer creates a new buffer (potential copy), so mutations of the
// transfer result do NOT affect the clone source.
//
// In this test:
// - Clone result is used by transfer (creates new buffer)
// - Fill mutates the TRANSFER result, not the clone result
// - Therefore clone can be elided since clone result is never mutated

// CHECK-LABEL: @transfer_breaks_aliasing_chain
// CHECK-SAME: (%[[BUFFER:.+]]: !util.buffer)
util.func public @transfer_breaks_aliasing_chain(%arg0: !util.buffer) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c123_i32 = arith.constant 123 : i32

  // Import WITHOUT consume - this is a borrowed buffer.
  // CHECK: %[[IMPORTED:.+]] = stream.tensor.import %[[BUFFER]] :
  %imported = stream.tensor.import %arg0 : !util.buffer -> tensor<64xf32> in !stream.resource<external>{%c256}

  // Clone CAN be elided because:
  // 1. Even though source is borrowed, clone RESULT is never mutated
  // 2. Transfer creates a new buffer, so fill mutates transfer's buffer, not clone's
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %imported : !stream.resource<external>{%c256} -> !stream.resource<external>{%c256}

  // Transfer creates a new buffer (not a pass-through/tied op).
  // Same-type same-affinity transfer can also be elided.
  // CHECK-NOT: stream.async.transfer
  %transfer = stream.async.transfer %clone : !stream.resource<external>{%c256} -> !stream.resource<external>{%c256}

  // Fill mutates the transfer result, which is the import buffer after elision.
  // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[IMPORTED]]
  %fill = stream.async.fill %c123_i32, %transfer[%c0 to %c256 for %c256] : i32 -> %transfer as !stream.resource<external>{%c256}

  util.return %fill : !stream.resource<external>
}
