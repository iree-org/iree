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

// -----

stream.executable private @ex {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%binding: !stream.binding) {
      return
    }
  }
}

// Tests scf.if where clone is mutated inside branch, but source is used after.
// Clone must be preserved to protect source for the post-if dispatch.

// CHECK-LABEL: @scf_if_clone_mutated_source_used_after
util.func public @scf_if_clone_mutated_source_used_after(%cond: i1, %size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SOURCE:.+]] = stream.async.splat
  %source = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[IF_RESULT:.+]] = scf.if
  %if_result = scf.if %cond -> !stream.resource<*> {
    // Clone preserved because source is used after the scf.if.
    // CHECK: %[[CLONE:.+]] = stream.async.clone %[[SOURCE]]
    %clone = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
    // CHECK: stream.async.dispatch @ex::@dispatch(%[[CLONE]]{{.*}}) : ({{.*}}) -> %[[CLONE]]
    %mutated = stream.async.dispatch @ex::@dispatch(%clone[%c0 to %size for %size])
        : (!stream.resource<*>{%size}) -> %clone{%size}
    scf.yield %mutated : !stream.resource<*>
  } else {
    // CHECK: stream.async.dispatch @ex::@dispatch(%[[SOURCE]]
    %read = stream.async.dispatch @ex::@dispatch(%source[%c0 to %size for %size])
        : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
    scf.yield %read : !stream.resource<*>
  }
  // Post-if use of source - this is why the clone must be preserved.
  // CHECK: stream.async.dispatch @ex::@dispatch(%[[SOURCE]]
  %after = stream.async.dispatch @ex::@dispatch(%source[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // CHECK: util.return %[[IF_RESULT]]
  util.return %if_result, %after : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests scf.for where clone is mutated but source is used after the loop.
// Clone must be preserved to protect source for the post-loop dispatch.

// CHECK-LABEL: @scf_for_clone_mutated_source_used_after
util.func public @scf_for_clone_mutated_source_used_after(%size: index, %count: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SOURCE:.+]] = stream.async.splat
  %source = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // Clone preserved because source is used after the loop.
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[SOURCE]]
  %clone = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[FOR:.+]] = scf.for {{.*}} iter_args(%[[ITER:.+]] = %[[CLONE]])
  %loop_result = scf.for %i = %c0 to %count step %c1 iter_args(%iter = %clone) -> !stream.resource<*> {
    // CHECK: stream.async.fill {{.*}}, %[[ITER]]
    %fill = stream.async.fill %c123_i32, %iter[%c0 to %size for %size] : i32 -> %iter as !stream.resource<*>{%size}
    scf.yield %fill : !stream.resource<*>
  }
  // Post-loop use of source - this is why the clone must be preserved.
  // CHECK: stream.async.dispatch @ex::@dispatch(%[[SOURCE]]
  %after = stream.async.dispatch @ex::@dispatch(%source[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // CHECK: util.return %[[FOR]]
  util.return %loop_result, %after : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests scf.if where both branches mutate clones, and source is used after.
// Both clones must be preserved to protect source for the post-if dispatch.

// CHECK-LABEL: @scf_if_both_branches_mutate_source_used_after
util.func public @scf_if_both_branches_mutate_source_used_after(%cond: i1, %size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  %c456_i32 = arith.constant 456 : i32
  // CHECK: %[[SOURCE:.+]] = stream.async.splat
  %source = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[IF_RESULT:.+]] = scf.if
  %if_result = scf.if %cond -> !stream.resource<*> {
    // CHECK: %[[CLONE1:.+]] = stream.async.clone %[[SOURCE]]
    %clone = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
    // CHECK: stream.async.fill %c123_i32, %[[CLONE1]]
    %fill = stream.async.fill %c123_i32, %clone[%c0 to %size for %size] : i32 -> %clone as !stream.resource<*>{%size}
    scf.yield %fill : !stream.resource<*>
  } else {
    // CHECK: %[[CLONE2:.+]] = stream.async.clone %[[SOURCE]]
    %clone = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
    // CHECK: stream.async.fill %c456_i32, %[[CLONE2]]
    %fill = stream.async.fill %c456_i32, %clone[%c0 to %size for %size] : i32 -> %clone as !stream.resource<*>{%size}
    scf.yield %fill : !stream.resource<*>
  }
  // Post-if use of source - this is why both clones must be preserved.
  // CHECK: stream.async.dispatch @ex::@dispatch(%[[SOURCE]]
  %after = stream.async.dispatch @ex::@dispatch(%source[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // CHECK: util.return %[[IF_RESULT]]
  util.return %if_result, %after : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests scf.while where clone is mutated in the loop body, and source is used
// after the while loop. Clone must be preserved.

// CHECK-LABEL: @scf_while_clone_mutated_source_used_after
util.func public @scf_while_clone_mutated_source_used_after(%cond: i1, %size: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SOURCE:.+]] = stream.async.splat
  %source = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // Clone preserved because source is used after the while loop.
  // CHECK: %[[CLONE:.+]] = stream.async.clone %[[SOURCE]]
  %clone = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[WHILE:.+]] = scf.while
  %result = scf.while (%arg = %clone) : (!stream.resource<*>) -> !stream.resource<*> {
    scf.condition(%cond) %arg : !stream.resource<*>
  } do {
  ^bb0(%body_arg: !stream.resource<*>):
    // Mutation in the loop body.
    %fill = stream.async.fill %c123_i32, %body_arg[%c0 to %size for %size] : i32 -> %body_arg as !stream.resource<*>{%size}
    scf.yield %fill : !stream.resource<*>
  }
  // Post-while use of source.
  // CHECK: stream.async.dispatch @ex::@dispatch(%[[SOURCE]]
  %after = stream.async.dispatch @ex::@dispatch(%source[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // CHECK: util.return %[[WHILE]]
  util.return %result, %after : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests that a read-only clone in a loop is elided even when source is used
// after the loop. Since neither source nor clone result is ever mutated, the
// copy is unnecessary.

// CHECK-LABEL: @scf_for_readonly_clone_elided
util.func public @scf_for_readonly_clone_elided(%size: index, %count: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SOURCE:.+]] = stream.async.splat
  %source = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // Clone is elided: neither source nor clone result is ever mutated.
  // CHECK-NOT: stream.async.clone
  %clone = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
  // CHECK: %[[LOOP_RESULT:.+]] = scf.for {{.*}} iter_args({{.*}} = %[[SOURCE]])
  %loop_result = scf.for %i = %c0 to %count step %c1 iter_args(%acc = %clone) -> !stream.resource<*> {
    // Dispatch reads %acc but produces a new resource (no tied output to %acc).
    %dispatched = stream.async.dispatch @ex::@dispatch(%acc[%c0 to %size for %size])
        : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
    scf.yield %dispatched : !stream.resource<*>
  }
  // Consume loop_result with a non-tied dispatch so the clone's resource
  // doesn't escape through the function return.
  // CHECK: stream.async.dispatch @ex::@dispatch(%[[LOOP_RESULT]]
  %consumed = stream.async.dispatch @ex::@dispatch(%loop_result[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // Post-loop read of source.
  // CHECK: stream.async.dispatch @ex::@dispatch(%[[SOURCE]]
  %after = stream.async.dispatch @ex::@dispatch(%source[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  util.return %consumed, %after : !stream.resource<*>, !stream.resource<*>
}

// -----

// Tests nested scf.for inside scf.if where clone is mutated in the loop,
// and source is used after the outer scf.if. Clone must be preserved.

// CHECK-LABEL: @scf_nested_for_in_if_source_used_after
util.func public @scf_nested_for_in_if_source_used_after(%cond: i1, %size: index, %count: index) -> (!stream.resource<*>, !stream.resource<*>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c123_i32 = arith.constant 123 : i32
  // CHECK: %[[SOURCE:.+]] = stream.async.splat
  %source = stream.async.splat %c123_i32 : i32 -> !stream.resource<*>{%size}
  // CHECK: %[[IF_RESULT:.+]] = scf.if
  %if_result = scf.if %cond -> !stream.resource<*> {
    // Clone inside if is preserved because source is used after the scf.if.
    // CHECK: %[[CLONE:.+]] = stream.async.clone %[[SOURCE]]
    %clone = stream.async.clone %source : !stream.resource<*>{%size} -> !stream.resource<*>{%size}
    // Mutating loop over the clone.
    // CHECK: scf.for {{.*}} iter_args({{.*}} = %[[CLONE]])
    %loop_result = scf.for %i = %c0 to %count step %c1 iter_args(%iter = %clone) -> !stream.resource<*> {
      %fill = stream.async.fill %c123_i32, %iter[%c0 to %size for %size] : i32 -> %iter as !stream.resource<*>{%size}
      scf.yield %fill : !stream.resource<*>
    }
    scf.yield %loop_result : !stream.resource<*>
  } else {
    scf.yield %source : !stream.resource<*>
  }
  // Post-if use of source.
  // CHECK: stream.async.dispatch @ex::@dispatch(%[[SOURCE]]
  %after = stream.async.dispatch @ex::@dispatch(%source[%c0 to %size for %size])
      : (!stream.resource<*>{%size}) -> !stream.resource<*>{%size}
  // CHECK: util.return %[[IF_RESULT]]
  util.return %if_result, %after : !stream.resource<*>, !stream.resource<*>
}
