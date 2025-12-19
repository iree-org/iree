// RUN: iree-opt --split-input-file --iree-stream-elide-async-copies --cse %s | FileCheck %s

// Tests scf.if with conditional in-place fill (no clone needed).
// The resource can be conditionally mutated without cloning since it's the last use.

// CHECK-LABEL: util.func private @scf_if_conditional_inplace_fill_callee
// CHECK-SAME: (%[[COND:.+]]: i1, %[[RESOURCE:.+]]: !stream.resource<*>, %{{.+}}: index)
util.func private @scf_if_conditional_inplace_fill_callee(%cond: i1, %resource: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[RESULT:.+]] = scf.if %[[COND]]
  %result = scf.if %cond -> !stream.resource<*> {
    // CHECK-NOT: stream.async.clone
    %clone = stream.async.clone %resource : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
    // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[RESOURCE]]
    %fill = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %0 as !stream.resource<*>{%size}
    // CHECK: scf.yield %[[FILL]]
    scf.yield %fill : !stream.resource<*>
  } else {
    // CHECK: scf.yield %[[RESOURCE]]
    scf.yield %resource : !stream.resource<*>
  }
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}
// CHECK-LABEL: @scf_if_conditional_inplace_fill_caller
util.func public @scf_if_conditional_inplace_fill_caller(%cond: i1, %size: index) -> !stream.resource<*> {
  %c123_i32 = arith.constant 123 : i32
  %initial = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  %result = util.call @scf_if_conditional_inplace_fill_callee(%cond, %initial, %size) : (i1, !stream.resource<*>, index) -> !stream.resource<*>
  util.return %result : !stream.resource<*>
}

// -----


// Tests scf.if with in-place operations in both branches.
// Both branches can mutate the resource in-place without cloning.

// CHECK-LABEL: util.func private @scf_if_both_branches_inplace_callee
// CHECK-SAME: (%[[COND:.+]]: i1, %[[RESOURCE:.+]]: !stream.resource<*>, %{{.+}}: index)
util.func private @scf_if_both_branches_inplace_callee(%cond: i1, %resource: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  // CHECK: %[[RESULT:.+]] = scf.if %[[COND]]
  %result = scf.if %cond -> !stream.resource<*> {
    // CHECK-NOT: stream.async.clone
    %clone = stream.async.clone %resource : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
    // CHECK: %[[FILL1:.+]] = stream.async.fill %c123_i32, %[[RESOURCE]]
    %fill = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %0 as !stream.resource<*>{%size}
    // CHECK: scf.yield %[[FILL1]]
    scf.yield %fill : !stream.resource<*>
  } else {
    // CHECK-NOT: stream.async.clone
    %clone = stream.async.clone %resource : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
    // CHECK: %[[FILL2:.+]] = stream.async.fill %c456_i32, %[[RESOURCE]]
    %fill = stream.async.fill %c456_i32, %clone[%c0 to %c100 for %c100] : i32 -> %0 as !stream.resource<*>{%size}
    // CHECK: scf.yield %[[FILL2]]
    scf.yield %fill : !stream.resource<*>
  }
  // CHECK: util.return %[[RESULT]]
  util.return %result : !stream.resource<*>
}
// CHECK-LABEL: @scf_if_both_branches_inplace_caller
util.func public @scf_if_both_branches_inplace_caller(%cond: i1, %size: index) -> !stream.resource<*> {
  %c123_i32 = arith.constant 123 : i32
  %initial = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  %result = util.call @scf_if_both_branches_inplace_callee(%cond, %initial, %size) : (i1, !stream.resource<*>, index) -> !stream.resource<*>
  util.return %result : !stream.resource<*>
}

// -----

// Tests scf.for with in-place operations in loop body (sequence a -> b -> b -> b).
// Loop iter_arg can be mutated in-place without cloning since it's passed by-value.

// CHECK-LABEL: util.func private @scf_for_inplace_sequence_callee
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>, %{{.+}}: index, %[[COUNT:.+]]: index)
util.func private @scf_for_inplace_sequence_callee(%resource: !stream.resource<*>, %size: index, %count: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[FOR:.+]] = scf.for %{{.+}} = %c0 to %[[COUNT]] step %c1 iter_args(%[[ITER:.+]] = %[[RESOURCE]])
  %result = scf.for %i = %c0 to %count step %c1 iter_args(%iter = %resource) -> !stream.resource<*> {
    // CHECK-NOT: stream.async.clone
    %clone = stream.async.clone %iter : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
    // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[ITER]]
    %fill = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %0 as !stream.resource<*>{%size}
    // CHECK: scf.yield %[[FILL]]
    scf.yield %fill : !stream.resource<*>
  }
  // CHECK: util.return %[[FOR]]
  util.return %result : !stream.resource<*>
}
// CHECK-LABEL: @scf_for_inplace_sequence_caller
util.func public @scf_for_inplace_sequence_caller(%size: index, %count: index) -> !stream.resource<*> {
  %c123_i32 = arith.constant 123 : i32
  %initial = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  %result = util.call @scf_for_inplace_sequence_callee(%initial, %size, %count) : (!stream.resource<*>, index, index) -> !stream.resource<*>
  util.return %result : !stream.resource<*>
}

// -----

// Tests scf.for with chained in-place operations (a -> b -> b -> b -> c).
// Multiple operations per iteration can all mutate in-place.

// CHECK-LABEL: util.func private @scf_for_chained_inplace_callee
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>, %{{.+}}: index, %[[COUNT:.+]]: index)
util.func private @scf_for_chained_inplace_callee(%resource: !stream.resource<*>, %size: index, %count: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c50 = arith.constant 50 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  // CHECK: %[[FOR:.+]] = scf.for %{{.+}} = %c0 to %[[COUNT]] step %c1 iter_args(%[[ITER:.+]] = %[[RESOURCE]])
  %result = scf.for %i = %c0 to %count step %c1 iter_args(%iter = %resource) -> !stream.resource<*> {
    // First operation: clone elided.
    // CHECK-NOT: stream.async.clone
    %clone1 = stream.async.clone %iter : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
    // CHECK: %[[FILL1:.+]] = stream.async.fill %c123_i32, %[[ITER]]
    %fill1 = stream.async.fill %c123_i32, %clone1[%c0 to %c50 for %c50] : i32 -> %0 as !stream.resource<*>{%size}

    // Second operation: clone elided, operates on result of first.
    // CHECK-NOT: stream.async.clone
    %clone2 = stream.async.clone %fill1 : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
    // CHECK: %[[FILL2:.+]] = stream.async.fill %c456_i32, %[[FILL1]]
    %fill2 = stream.async.fill %c456_i32, %clone2[%c50 to %c100 for %c50] : i32 -> %2 as !stream.resource<*>{%size}

    // CHECK: scf.yield %[[FILL2]]
    scf.yield %fill2 : !stream.resource<*>
  }
  // CHECK: util.return %[[FOR]]
  util.return %result : !stream.resource<*>
}
// CHECK-LABEL: @scf_for_chained_inplace_caller
util.func public @scf_for_chained_inplace_caller(%size: index, %count: index) -> !stream.resource<*> {
  %c123_i32 = arith.constant 123 : i32
  %initial = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  %result = util.call @scf_for_chained_inplace_callee(%initial, %size, %count) : (!stream.resource<*>, index, index) -> !stream.resource<*>
  util.return %result : !stream.resource<*>
}

// -----

// Tests scf.while with in-place operations in both before and after regions.
// Demonstrates loop arguments can be mutated in-place throughout.

// CHECK-LABEL: util.func private @scf_while_inplace_both_regions_callee
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>, %{{.+}}: index)
util.func private @scf_while_inplace_both_regions_callee(%resource: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c50 = arith.constant 50 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32

  // CHECK: %[[WHILE:.+]]:2 = scf.while (%[[ITER:.+]] = %[[RESOURCE]], %[[COUNTER:.+]] = %c1_i32)
  %result:2 = scf.while (%iter = %resource, %counter = %c1_i32) : (!stream.resource<*>, i32) -> (!stream.resource<*>, i32) {
    // Before region: check condition and mutate.
    %cmp = arith.cmpi slt, %counter, %c10_i32 : i32
    // CHECK-NOT: stream.async.clone
    %clone = stream.async.clone %iter : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
    // CHECK: %[[FILL_BEFORE:.+]] = stream.async.fill %c123_i32, %[[ITER]]
    %fill = stream.async.fill %c123_i32, %clone[%c0 to %c50 for %c50] : i32 -> %0 as !stream.resource<*>{%size}
    // CHECK: scf.condition(%{{.+}}) %[[FILL_BEFORE]], %[[COUNTER]]
    scf.condition(%cmp) %fill, %counter : !stream.resource<*>, i32
  } do {
  ^bb0(%iter_after: !stream.resource<*>, %counter_after: i32):
    // After region: mutate again and increment.
    // CHECK: %[[INC:.+]] = arith.addi
    %inc = arith.addi %counter_after, %c1_i32 : i32
    // CHECK-NOT: stream.async.clone
    %clone = stream.async.clone %iter_after : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
    // CHECK: %[[FILL_AFTER:.+]] = stream.async.fill %c456_i32
    %fill = stream.async.fill %c456_i32, %clone[%c50 to %c100 for %c50] : i32 -> %0 as !stream.resource<*>{%size}
    // CHECK: scf.yield %[[FILL_AFTER]], %[[INC]]
    scf.yield %fill, %inc : !stream.resource<*>, i32
  }
  // CHECK: util.return %[[WHILE]]#0
  util.return %result#0 : !stream.resource<*>
}
// CHECK-LABEL: @scf_while_inplace_both_regions_caller
util.func public @scf_while_inplace_both_regions_caller(%size: index) -> !stream.resource<*> {
  %c123_i32 = arith.constant 123 : i32
  %initial = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  %result = util.call @scf_while_inplace_both_regions_callee(%initial, %size) : (!stream.resource<*>, index) -> !stream.resource<*>
  util.return %result : !stream.resource<*>
}

// -----

// Tests nested scf.for with in-place operations (demonstrates deep nesting).
// Inner and outer loops can both mutate without cloning.

// CHECK-LABEL: util.func private @scf_nested_for_inplace_callee
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>, %{{.+}}: index)
util.func private @scf_nested_for_inplace_callee(%resource: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c100 = arith.constant 100 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32

  // CHECK: %[[OUTER:.+]] = scf.for %{{.+}} = %c0 to %c5 step %c1 iter_args(%[[OUTER_ITER:.+]] = %[[RESOURCE]])
  %outer = scf.for %i = %c0 to %c5 step %c1 iter_args(%outer_iter = %resource) -> !stream.resource<*> {
    // CHECK: %[[INNER:.+]] = scf.for %{{.+}} = %c0 to %c5 step %c1 iter_args(%[[INNER_ITER:.+]] = %[[OUTER_ITER]])
    %inner = scf.for %j = %c0 to %c5 step %c1 iter_args(%inner_iter = %outer_iter) -> !stream.resource<*> {
      // CHECK-NOT: stream.async.clone
      %clone = stream.async.clone %inner_iter : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
      // CHECK: %[[FILL:.+]] = stream.async.fill %c123_i32, %[[INNER_ITER]]
      %fill = stream.async.fill %c123_i32, %clone[%c0 to %c100 for %c100] : i32 -> %0 as !stream.resource<*>{%size}
      // CHECK: scf.yield %[[FILL]]
      scf.yield %fill : !stream.resource<*>
    }
    // CHECK: scf.yield %[[INNER]]
    scf.yield %inner : !stream.resource<*>
  }
  // CHECK: util.return %[[OUTER]]
  util.return %outer : !stream.resource<*>
}
// CHECK-LABEL: @scf_nested_for_inplace_caller
util.func public @scf_nested_for_inplace_caller(%size: index) -> !stream.resource<*> {
  %c123_i32 = arith.constant 123 : i32
  %initial = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  %result = util.call @scf_nested_for_inplace_callee(%initial, %size) : (!stream.resource<*>, index) -> !stream.resource<*>
  util.return %result : !stream.resource<*>
}
