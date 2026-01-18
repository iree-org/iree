// RUN: iree-opt --split-input-file --iree-stream-automatic-reference-counting %s | FileCheck %s

// Tests that resources allocated outside a loop and captured inside have their
// lifetime extended (NOT marked indeterminate).

// CHECK-LABEL: @loop_captured_resource
// CHECK-SAME: (%[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index)
util.func private @loop_captured_resource(%input_tp: !stream.timepoint, %size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TP:.+]] = stream.resource.alloca uninitialized await(%[[INPUT_TP]]) => !stream.resource<transient>{%[[SIZE]]}
  %resource, %alloca_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint

  // Loop captures and uses resource (cmd-level pattern).
  // CHECK: %[[LOOP_RESULT:.+]] = scf.for
  %loop_result = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %alloca_tp) -> !stream.timepoint {
    // CHECK: stream.test.timeline_op await(%{{.+}})
    %cmd_tp = stream.test.timeline_op await(%arg) =>
      with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    scf.yield %cmd_tp : !stream.timepoint
  }

  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[ALLOCA_TP]], %[[LOOP_RESULT]])
  // CHECK: %[[DEALLOCA_TP:.+]] = stream.resource.dealloca origin await(%[[JOIN]]) => %[[RESOURCE]]
  // CHECK-NOT: marked indeterminate
  // CHECK: util.return %[[DEALLOCA_TP]]
  util.return %loop_result : !stream.timepoint
}

// -----

// Tests that resources allocated INSIDE a loop that never escape can be
// deallocated inside the loop body (local lifetime).

// CHECK-LABEL: @loop_local_resource
util.func private @loop_local_resource(%input_tp: !stream.timepoint, %size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // CHECK: scf.for
  %loop_result = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %input_tp) -> !stream.timepoint {
    // CHECK: %[[LOCAL_RESOURCE:.+]], %[[LOCAL_ALLOCA_TP:.+]] = stream.resource.alloca
    %local_resource, %local_alloca_tp = stream.resource.alloca uninitialized await(%arg) => !stream.resource<transient>{%size} => !stream.timepoint

    // CHECK: %[[CMD_TP:.+]] = stream.test.timeline_op
    %cmd_tp = stream.test.timeline_op await(%local_alloca_tp) =>
      with(%local_resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint

    // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[LOCAL_ALLOCA_TP]], %[[CMD_TP]])
    // CHECK: %[[LOCAL_DEALLOCA_TP:.+]] = stream.resource.dealloca origin await(%[[JOIN]]) => %[[LOCAL_RESOURCE]]
    // CHECK: scf.yield %[[LOCAL_DEALLOCA_TP]]
    // Local resource deallocated inside loop body (never escapes).
    scf.yield %cmd_tp : !stream.timepoint
  }

  util.return %loop_result : !stream.timepoint
}

// -----

// Tests scf.if with captured resource in both branches.

// CHECK-LABEL: @if_captured_resource
// CHECK-SAME: ({{.+}}: i1, %[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index)
util.func private @if_captured_resource(%cond: i1, %input_tp: !stream.timepoint, %size: index) -> !stream.timepoint {
  %c1 = arith.constant 1 : index

  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TP:.+]] = stream.resource.alloca uninitialized await(%[[INPUT_TP]]) => !stream.resource<transient>{%[[SIZE]]}
  %resource, %alloca_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint

  // CHECK: %[[IF_RESULT:.+]] = scf.if
  %if_result = scf.if %cond -> !stream.timepoint {
    // CHECK: stream.test.timeline_op await(%[[ALLOCA_TP]])
    %then_tp = stream.test.timeline_op await(%alloca_tp) =>
      with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    scf.yield %then_tp : !stream.timepoint
  } else {
    // CHECK: stream.test.timeline_op await(%[[ALLOCA_TP]])
    %else_tp = stream.test.timeline_op await(%alloca_tp) =>
      with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    scf.yield %else_tp : !stream.timepoint
  }

  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[ALLOCA_TP]], %[[IF_RESULT]])
  // CHECK: %[[DEALLOCA_TP:.+]] = stream.resource.dealloca origin await(%[[JOIN]]) => %[[RESOURCE]]
  // CHECK-NOT: marked indeterminate
  // CHECK: util.return %[[DEALLOCA_TP]]
  util.return %if_result : !stream.timepoint
}

// -----

// Tests scf.if with local resource in then-branch that doesn't escape.

// CHECK-LABEL: @if_local_resource
// CHECK-SAME: ({{.+}}: i1, %[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index)
util.func private @if_local_resource(%cond: i1, %input_tp: !stream.timepoint, %size: index) -> !stream.timepoint {
  %c1 = arith.constant 1 : index

  // CHECK: scf.if
  %if_result = scf.if %cond -> !stream.timepoint {
    // CHECK: %[[LOCAL_RESOURCE:.+]], %[[LOCAL_ALLOCA_TP:.+]] = stream.resource.alloca uninitialized await(%[[INPUT_TP]]) => !stream.resource<transient>{%[[SIZE]]}
    %local_resource, %local_alloca_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint

    // CHECK: %[[THEN_TP:.+]] = stream.test.timeline_op await(%[[LOCAL_ALLOCA_TP]])
    %then_tp = stream.test.timeline_op await(%local_alloca_tp) =>
      with(%local_resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint

    // CHECK: %[[LOCAL_DEALLOCA_TP:.+]] = stream.resource.dealloca origin await(%[[THEN_TP]]) => %[[LOCAL_RESOURCE]]
    // CHECK: scf.yield %[[LOCAL_DEALLOCA_TP]]
    // Local resource deallocated inside then-branch (coverage analysis eliminates redundant join).
    scf.yield %then_tp : !stream.timepoint
  } else {
    scf.yield %input_tp : !stream.timepoint
  }

  util.return %if_result : !stream.timepoint
}

// -----

// Tests nested control flow: scf.if inside scf.for with captured resource.

// CHECK-LABEL: @nested_if_in_loop
// CHECK-SAME: ({{.+}}: i1, %[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index)
util.func private @nested_if_in_loop(%cond: i1, %input_tp: !stream.timepoint, %size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TP:.+]] = stream.resource.alloca uninitialized await(%[[INPUT_TP]]) => !stream.resource<transient>{%[[SIZE]]}
  %resource, %alloca_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint

  // CHECK: %[[LOOP_RESULT:.+]] = scf.for
  %loop_result = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %alloca_tp) -> !stream.timepoint {
    // CHECK: scf.if
    %if_tp = scf.if %cond -> !stream.timepoint {
      // CHECK: stream.test.timeline_op
      %then_tp = stream.test.timeline_op await(%arg) =>
        with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
      scf.yield %then_tp : !stream.timepoint
    } else {
      // CHECK: stream.test.timeline_op
      %else_tp = stream.test.timeline_op await(%arg) =>
        with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
      scf.yield %else_tp : !stream.timepoint
    }
    scf.yield %if_tp : !stream.timepoint
  }

  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[ALLOCA_TP]], %[[LOOP_RESULT]])
  // CHECK: %[[DEALLOCA_TP:.+]] = stream.resource.dealloca origin await(%[[JOIN]]) => %[[RESOURCE]]
  // Captured resource through nested if-in-loop should NOT be indeterminate.
  // CHECK-NOT: marked indeterminate
  // CHECK: util.return %[[DEALLOCA_TP]]
  util.return %loop_result : !stream.timepoint
}

// -----

// Tests nested control flow: scf.for inside scf.if with captured resource.

// CHECK-LABEL: @nested_loop_in_if
// CHECK-SAME: ({{.+}}: i1, %[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index)
util.func private @nested_loop_in_if(%cond: i1, %input_tp: !stream.timepoint, %size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TP:.+]] = stream.resource.alloca uninitialized await(%[[INPUT_TP]]) => !stream.resource<transient>{%[[SIZE]]}
  %resource, %alloca_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint

  // CHECK: %[[IF_RESULT:.+]] = scf.if
  %if_result = scf.if %cond -> !stream.timepoint {
    // CHECK: scf.for
    %loop_result = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %alloca_tp) -> !stream.timepoint {
      // CHECK: stream.test.timeline_op
      %cmd_tp = stream.test.timeline_op await(%arg) =>
        with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
      scf.yield %cmd_tp : !stream.timepoint
    }
    scf.yield %loop_result : !stream.timepoint
  } else {
    scf.yield %alloca_tp : !stream.timepoint
  }

  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[ALLOCA_TP]], %[[IF_RESULT]])
  // CHECK: %[[DEALLOCA_TP:.+]] = stream.resource.dealloca origin await(%[[JOIN]]) => %[[RESOURCE]]
  // Captured resource through nested loop-in-if should NOT be indeterminate.
  // CHECK-NOT: marked indeterminate
  // CHECK: util.return %[[DEALLOCA_TP]]
  util.return %if_result : !stream.timepoint
}

// -----

// Tests multiple captured resources in a loop.

// CHECK-LABEL: @loop_multiple_captured
// CHECK-SAME: (%[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE1:.+]]: index, %[[SIZE2:.+]]: index)
util.func private @loop_multiple_captured(%input_tp: !stream.timepoint, %size1: index, %size2: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // CHECK: %[[RESOURCE1:.+]], %[[ALLOCA_TP1:.+]] = stream.resource.alloca uninitialized await(%[[INPUT_TP]]) => !stream.resource<transient>{%[[SIZE1]]}
  %resource1, %alloca_tp1 = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size1} => !stream.timepoint

  // CHECK: %[[RESOURCE2:.+]], %[[ALLOCA_TP2:.+]] = stream.resource.alloca uninitialized await(%[[ALLOCA_TP1]]) => !stream.resource<transient>{%[[SIZE2]]}
  %resource2, %alloca_tp2 = stream.resource.alloca uninitialized await(%alloca_tp1) => !stream.resource<transient>{%size2} => !stream.timepoint

  // CHECK: %[[LOOP_RESULT:.+]] = scf.for
  %loop_result = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %alloca_tp2) -> !stream.timepoint {
    // CHECK: stream.test.timeline_op
    %cmd_tp = stream.test.timeline_op await(%arg) =>
      with(%resource1, %resource2) : (!stream.resource<transient>{%size1}, !stream.resource<transient>{%size2}) -> () => !stream.timepoint
    scf.yield %cmd_tp : !stream.timepoint
  }

  // CHECK: %[[JOIN1:.+]] = stream.timepoint.join max(%[[ALLOCA_TP1]], %[[LOOP_RESULT]])
  // CHECK: %[[DEALLOCA1:.+]] = stream.resource.dealloca origin await(%[[JOIN1]]) => %[[RESOURCE1]]
  // CHECK: %[[JOIN2:.+]] = stream.timepoint.join max(%[[ALLOCA_TP2]], %[[DEALLOCA1]])
  // CHECK: %[[DEALLOCA2:.+]] = stream.resource.dealloca origin await(%[[JOIN2]]) => %[[RESOURCE2]]
  // Both captured resources should NOT be indeterminate.
  // CHECK-NOT: marked indeterminate
  // CHECK: util.return %[[DEALLOCA2]]
  util.return %loop_result : !stream.timepoint
}

// -----

// Tests scf.for with iter_args carrying a resource (rare case).

// CHECK-LABEL: @loop_iter_args_resource
util.func private @loop_iter_args_resource(%input_tp: !stream.timepoint, %initial_resource: !stream.resource<transient>, %size: index) -> (!stream.resource<transient>, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // CHECK: %[[RESULT:.+]]:2 = scf.for
  %result_resource, %result_tp = scf.for %i = %c0 to %c10 step %c1
      iter_args(%iter_resource = %initial_resource, %iter_tp = %input_tp) -> (!stream.resource<transient>, !stream.timepoint) {
    // CHECK: stream.test.timeline_op
    %cmd_tp = stream.test.timeline_op await(%iter_tp) =>
      with(%iter_resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    scf.yield %iter_resource, %cmd_tp : !stream.resource<transient>, !stream.timepoint
  }

  // Loop-carried resource via iter_args should be aliased correctly.
  // CHECK-NOT: marked indeterminate
  // CHECK: util.return %[[RESULT]]#0, %[[RESULT]]#1
  util.return %result_resource, %result_tp : !stream.resource<transient>, !stream.timepoint
}

// -----

// Tests deeply nested control flow (3 levels).

// CHECK-LABEL: @deeply_nested
// CHECK-SAME: ({{.+}}: i1, {{.+}}: i1, %[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index)
util.func private @deeply_nested(%cond1: i1, %cond2: i1, %input_tp: !stream.timepoint, %size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index

  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TP:.+]] = stream.resource.alloca uninitialized await(%[[INPUT_TP]]) => !stream.resource<transient>{%[[SIZE]]}
  %resource, %alloca_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint

  // CHECK: %[[IF1_RESULT:.+]] = scf.if
  %if1_result = scf.if %cond1 -> !stream.timepoint {
    // CHECK: scf.for
    %loop_result = scf.for %i = %c0 to %c5 step %c1 iter_args(%arg = %alloca_tp) -> !stream.timepoint {
      // CHECK: scf.if
      %if2_result = scf.if %cond2 -> !stream.timepoint {
        // CHECK: stream.test.timeline_op
        %cmd_tp = stream.test.timeline_op await(%arg) =>
          with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
        scf.yield %cmd_tp : !stream.timepoint
      } else {
        scf.yield %arg : !stream.timepoint
      }
      scf.yield %if2_result : !stream.timepoint
    }
    scf.yield %loop_result : !stream.timepoint
  } else {
    scf.yield %alloca_tp : !stream.timepoint
  }

  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[ALLOCA_TP]], %[[IF1_RESULT]])
  // CHECK: %[[DEALLOCA_TP:.+]] = stream.resource.dealloca origin await(%[[JOIN]]) => %[[RESOURCE]]
  // Deeply nested captured resource should NOT be indeterminate.
  // CHECK-NOT: marked indeterminate
  // CHECK: util.return %[[DEALLOCA_TP]]
  util.return %if1_result : !stream.timepoint
}

// -----

// Tests that a resource allocated INSIDE a loop and yielded OUT should NOT be
// deallocated inside the loop body (use-after-free bug fix).

// CHECK-LABEL: @loop_local_resource_yielded
// CHECK-SAME: ({{.+}}: !stream.timepoint, %[[SIZE:.+]]: index, {{.+}}: !stream.resource<transient>)
util.func private @loop_local_resource_yielded(%input_tp: !stream.timepoint, %size: index, %init_resource: !stream.resource<transient>) -> (!stream.resource<transient>, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // CHECK: %[[LOOP_RESULT:.+]]:2 = scf.for
  %loop_resource, %loop_tp = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg_res = %init_resource, %arg_tp = %input_tp) -> (!stream.resource<transient>, !stream.timepoint) {
    // CHECK: %[[LOCAL_RESOURCE:.+]], %[[LOCAL_ALLOCA_TP:.+]] = stream.resource.alloca uninitialized await(%{{.+}}) => !stream.resource<transient>{%[[SIZE]]}
    %local_resource, %local_alloca_tp = stream.resource.alloca uninitialized await(%arg_tp) => !stream.resource<transient>{%size} => !stream.timepoint

    // CHECK: %[[CMD_TP:.+]] = stream.test.timeline_op await(%[[LOCAL_ALLOCA_TP]])
    %cmd_tp = stream.test.timeline_op await(%local_alloca_tp) =>
      with(%local_resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint

    // Resource is yielded out - should NOT be deallocated inside loop.
    // CHECK-NOT: stream.resource.dealloca
    // CHECK: scf.yield %[[LOCAL_RESOURCE]], %[[CMD_TP]]
    scf.yield %local_resource, %cmd_tp : !stream.resource<transient>, !stream.timepoint
  }

  // The yielded resource should be available here for use.
  // CHECK: util.return %[[LOOP_RESULT]]#0, %[[LOOP_RESULT]]#1
  util.return %loop_resource, %loop_tp : !stream.resource<transient>, !stream.timepoint
}

// -----

// Tests that a resource allocated INSIDE an if-branch and yielded OUT should
// NOT be deallocated inside the branch (use-after-free bug fix).

// CHECK-LABEL: @if_local_resource_yielded
// CHECK-SAME: ({{.+}}: i1, %[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index, {{.+}}: !stream.resource<transient>)
util.func private @if_local_resource_yielded(%cond: i1, %input_tp: !stream.timepoint, %size: index, %else_resource: !stream.resource<transient>) -> (!stream.resource<transient>, !stream.timepoint) {
  // CHECK: %[[IF_RESULT:.+]]:2 = scf.if
  %if_resource, %if_tp = scf.if %cond -> (!stream.resource<transient>, !stream.timepoint) {
    // CHECK: %[[LOCAL_RESOURCE:.+]], %[[LOCAL_ALLOCA_TP:.+]] = stream.resource.alloca uninitialized await(%[[INPUT_TP]]) => !stream.resource<transient>{%[[SIZE]]}
    %local_resource, %local_alloca_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint

    // CHECK: %[[CMD_TP:.+]] = stream.test.timeline_op await(%[[LOCAL_ALLOCA_TP]])
    %cmd_tp = stream.test.timeline_op await(%local_alloca_tp) =>
      with(%local_resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint

    // Resource is yielded out - should NOT be deallocated inside branch.
    // CHECK-NOT: stream.resource.dealloca
    // CHECK: scf.yield %[[LOCAL_RESOURCE]], %[[CMD_TP]]
    scf.yield %local_resource, %cmd_tp : !stream.resource<transient>, !stream.timepoint
  } else {
    // Else branch yields a different resource.
    scf.yield %else_resource, %input_tp : !stream.resource<transient>, !stream.timepoint
  }

  // The yielded resource should be available here for use.
  // CHECK: util.return %[[IF_RESULT]]#0, %[[IF_RESULT]]#1
  util.return %if_resource, %if_tp : !stream.resource<transient>, !stream.timepoint
}

// -----

// Tests that when scf.for returns MULTIPLE timepoints, the pass creates a join
// and uses it for tracking captured resource lifetimes.

// CHECK-LABEL: @loop_multiple_timepoint_results
// CHECK-SAME: (%[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index)
util.func private @loop_multiple_timepoint_results(%input_tp: !stream.timepoint, %size: index) -> (!stream.timepoint, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TP:.+]] = stream.resource.alloca uninitialized await(%[[INPUT_TP]]) => !stream.resource<transient>{%[[SIZE]]}
  %resource, %alloca_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint

  // Loop returns TWO timepoints and captures a resource.
  // CHECK: %[[LOOP_RESULTS:.+]]:2 = scf.for
  %loop_tp1, %loop_tp2 = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg1 = %alloca_tp, %arg2 = %alloca_tp) -> (!stream.timepoint, !stream.timepoint) {
    // CHECK: stream.test.timeline_op await(%{{.+}})
    %cmd_tp1 = stream.test.timeline_op await(%arg1) =>
      with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint

    // Second command also uses captured resource.
    // CHECK: stream.test.timeline_op await(%{{.+}})
    %cmd_tp2 = stream.test.timeline_op await(%arg2) =>
      with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint

    scf.yield %cmd_tp1, %cmd_tp2 : !stream.timepoint, !stream.timepoint
  }

  // The pass should create a JOIN of the two loop result timepoints.
  // CHECK: %[[LOOP_JOIN:.+]] = stream.timepoint.join max(%[[LOOP_RESULTS]]#0, %[[LOOP_RESULTS]]#1)

  // The captured resource needs to await BOTH the alloca and loop execution.
  // The pass creates another join combining alloca_tp with the loop join.
  // CHECK: %[[FINAL_JOIN:.+]] = stream.timepoint.join max(%[[ALLOCA_TP]], %[[LOOP_JOIN]])

  // The captured resource is deallocated awaiting the final join.
  // CHECK: stream.resource.dealloca origin await(%[[FINAL_JOIN]]) => %[[RESOURCE]]

  // CHECK: util.return %[[LOOP_RESULTS]]#0, %[[LOOP_RESULTS]]#1
  util.return %loop_tp1, %loop_tp2 : !stream.timepoint, !stream.timepoint
}

// -----

// Tests scf.while with captured resource tracking.

// CHECK-LABEL: @while_captured_resource
// CHECK-SAME: (%[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index, {{.+}}: index)
util.func private @while_captured_resource(%input_tp: !stream.timepoint, %size: index, %bound: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TP:.+]] = stream.resource.alloca uninitialized await(%[[INPUT_TP]]) => !stream.resource<transient>{%[[SIZE]]}
  %resource, %alloca_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint

  // CHECK: %[[WHILE_RESULT:.+]]:2 = scf.while
  %while_result:2 = scf.while (%iter = %c0, %tp = %alloca_tp) : (index, !stream.timepoint) -> (index, !stream.timepoint) {
    %cond = arith.cmpi slt, %iter, %bound : index
    scf.condition(%cond) %iter, %tp : index, !stream.timepoint
  } do {
  ^bb0(%iter: index, %tp: !stream.timepoint):
    // CHECK: stream.test.timeline_op await(%{{.+}})
    %cmd_tp = stream.test.timeline_op await(%tp) =>
      with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint

    %next_iter = arith.addi %iter, %c1 : index
    scf.yield %next_iter, %cmd_tp : index, !stream.timepoint
  }

  // The captured resource needs to await both alloca and while execution.
  // The pass creates a join of the alloca timepoint and while result timepoint.
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[ALLOCA_TP]], %[[WHILE_RESULT]]#1)

  // CHECK: %[[DEALLOCA_TP:.+]] = stream.resource.dealloca origin await(%[[JOIN]]) => %[[RESOURCE]]
  // CHECK-NOT: marked indeterminate
  // CHECK: util.return %[[DEALLOCA_TP]]
  util.return %while_result#1 : !stream.timepoint
}

// -----

// Tests that a resource yielded from one branch but not another is correctly
// handled (resource available from both branches, but only allocated in one).

// CHECK-LABEL: @if_resource_from_one_branch
// CHECK-SAME: ({{.+}}: i1, %[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index, {{.+}}: !stream.resource<transient>)
util.func private @if_resource_from_one_branch(%cond: i1, %input_tp: !stream.timepoint, %size: index, %fallback_resource: !stream.resource<transient>) -> (!stream.resource<transient>, !stream.timepoint) {
  // CHECK: %[[IF_RESULT:.+]]:2 = scf.if
  %if_resource, %if_tp = scf.if %cond -> (!stream.resource<transient>, !stream.timepoint) {
    // Then-branch allocates a new resource and yields it.
    // CHECK: %[[LOCAL_RESOURCE:.+]], %[[LOCAL_ALLOCA_TP:.+]] = stream.resource.alloca uninitialized await(%[[INPUT_TP]]) => !stream.resource<transient>{%[[SIZE]]}
    %local_resource, %local_alloca_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint

    // CHECK: %[[CMD_TP:.+]] = stream.test.timeline_op await(%[[LOCAL_ALLOCA_TP]])
    %cmd_tp = stream.test.timeline_op await(%local_alloca_tp) =>
      with(%local_resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint

    // Resource yielded from then-branch - should NOT be deallocated.
    // CHECK-NOT: stream.resource.dealloca
    // CHECK: scf.yield %[[LOCAL_RESOURCE]], %[[CMD_TP]]
    scf.yield %local_resource, %cmd_tp : !stream.resource<transient>, !stream.timepoint
  } else {
    // Else-branch yields the fallback resource (defined outside).
    scf.yield %fallback_resource, %input_tp : !stream.resource<transient>, !stream.timepoint
  }

  // The yielded resource should be available for use.
  // CHECK: util.return %[[IF_RESULT]]#0, %[[IF_RESULT]]#1
  util.return %if_resource, %if_tp : !stream.resource<transient>, !stream.timepoint
}

// -----

// Tests deeply nested SCF operations (scf.if inside scf.while inside scf.for).

// CHECK-LABEL: @deeply_nested_scf
// CHECK-SAME: ({{.+}}: i1, %[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index, {{.+}}: index)
util.func private @deeply_nested_scf(%cond: i1, %input_tp: !stream.timepoint, %size: index, %bound: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TP:.+]] = stream.resource.alloca uninitialized await(%[[INPUT_TP]]) => !stream.resource<transient>{%[[SIZE]]}
  %resource, %alloca_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint

  // Outer loop: scf.for
  // CHECK: %[[FOR_RESULT:.+]] = scf.for
  %for_result = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg_tp = %alloca_tp) -> !stream.timepoint {
    // Middle loop: scf.while
    // CHECK: %[[WHILE_RESULT:.+]]:2 = scf.while
    %while_result:2 = scf.while (%iter = %c0, %tp = %arg_tp) : (index, !stream.timepoint) -> (index, !stream.timepoint) {
      %cond_check = arith.cmpi slt, %iter, %bound : index
      scf.condition(%cond_check) %iter, %tp : index, !stream.timepoint
    } do {
    ^bb0(%iter: index, %tp: !stream.timepoint):
      // Inner conditional: scf.if
      // CHECK: %[[IF_RESULT:.+]] = scf.if
      %if_result = scf.if %cond -> !stream.timepoint {
        // CHECK: stream.test.timeline_op await(%{{.+}})
        %cmd_tp = stream.test.timeline_op await(%tp) =>
          with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
        scf.yield %cmd_tp : !stream.timepoint
      } else {
        scf.yield %tp : !stream.timepoint
      }
      // CHECK: %[[NEXT_ITER:.+]] = arith.addi
      %next_iter = arith.addi %iter, %c1 : index
      // CHECK: scf.yield %[[NEXT_ITER]], %[[IF_RESULT]]
      scf.yield %next_iter, %if_result : index, !stream.timepoint
    }
    // CHECK: scf.yield %[[WHILE_RESULT]]#1
    scf.yield %while_result#1 : !stream.timepoint
  }

  // Resource captured through 3 levels of nesting should be tracked correctly.
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[ALLOCA_TP]], %[[FOR_RESULT]])
  // CHECK: %[[DEALLOCA_TP:.+]] = stream.resource.dealloca origin await(%[[JOIN]]) => %[[RESOURCE]]
  // CHECK-NOT: marked indeterminate
  // CHECK: util.return %[[DEALLOCA_TP]]
  util.return %for_result : !stream.timepoint
}

// -----

// Tests that timepoint coverage correctly spans parent and nested scf.for regions.
// A loop body that joins a parent-scope timepoint with an iter_arg timepoint
// requires the coverage analysis to track timepoints across scope boundaries.
// This guards against incorrectly localizing coverage per-block.

// CHECK-LABEL: @cross_scope_for_await_parent
// CHECK-SAME: (%[[PARENT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index)
util.func private @cross_scope_for_await_parent(%parent_tp: !stream.timepoint, %size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // Allocate resource in parent scope.
  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TP:.+]] = stream.resource.alloca
  %resource, %alloca_tp = stream.resource.alloca uninitialized await(%parent_tp) => !stream.resource<transient>{%size} => !stream.timepoint

  // Nested loop awaits on parent_tp and alloca_tp.
  // If coverage were per-block, covers(parent_tp, nested_tp) would fail.
  // CHECK: %[[LOOP_RESULT:.+]] = scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[ITER:.+]] = %[[ALLOCA_TP]])
  %loop_result = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg = %alloca_tp) -> !stream.timepoint {
    // Join parent timepoint with iter_arg - tests cross-scope coverage tracking.
    // CHECK:   %[[JOINED:.+]] = stream.timepoint.join max(%[[PARENT_TP]], %[[ITER]])
    %joined_tp = stream.timepoint.join max(%parent_tp, %arg) => !stream.timepoint
    // CHECK:   %[[CMD_TP:.+]] = stream.test.timeline_op await(%[[JOINED]]) => with(%[[RESOURCE]]) : (!stream.resource<transient>{%[[SIZE]]})
    %cmd_tp = stream.test.timeline_op await(%joined_tp) =>
      with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    // CHECK:   scf.yield %[[CMD_TP]]
    scf.yield %cmd_tp : !stream.timepoint
  }

  // Coverage must correctly track that parent_tp is covered by loop_result.
  // CHECK: %[[JOINED_TP:.+]] = stream.timepoint.join max(%[[ALLOCA_TP]], %[[LOOP_RESULT]])
  // CHECK: %[[DEALLOCA:.+]] = stream.resource.dealloca origin await(%[[JOINED_TP]]) => %[[RESOURCE]]
  // CHECK: util.return %[[DEALLOCA]]
  util.return %loop_result : !stream.timepoint
}

// -----

// Tests that timepoint coverage correctly spans parent and nested scf.if regions.
// Both if-branches join a parent-scope timepoint with an alloca timepoint,
// requiring the coverage analysis to track timepoints across scope boundaries.

// CHECK-LABEL: @cross_scope_if_await_parent
// CHECK-SAME: (%[[PARENT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index, {{.+}}: i1)
util.func private @cross_scope_if_await_parent(%parent_tp: !stream.timepoint, %size: index, %cond: i1) -> !stream.timepoint {
  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TP:.+]] = stream.resource.alloca
  %resource, %alloca_tp = stream.resource.alloca uninitialized await(%parent_tp) => !stream.resource<transient>{%size} => !stream.timepoint

  // Nested if awaits on parent_tp.
  // CHECK: %[[IF_RESULT:.+]] = scf.if
  %if_result = scf.if %cond -> !stream.timepoint {
    // Then branch joins parent with alloca timepoint.
    // CHECK:   %[[THEN_JOINED:.+]] = stream.timepoint.join max(%[[PARENT_TP]], %[[ALLOCA_TP]])
    %then_joined_tp = stream.timepoint.join max(%parent_tp, %alloca_tp) => !stream.timepoint
    // CHECK:   %[[THEN_TP:.+]] = stream.test.timeline_op await(%[[THEN_JOINED]]) => with(%[[RESOURCE]]) : (!stream.resource<transient>{%[[SIZE]]})
    %then_tp = stream.test.timeline_op await(%then_joined_tp) =>
      with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    // CHECK:   scf.yield %[[THEN_TP]]
    scf.yield %then_tp : !stream.timepoint
  } else {
    // Else branch joins parent with alloca timepoint.
    // CHECK:   %[[ELSE_JOINED:.+]] = stream.timepoint.join max(%[[PARENT_TP]], %[[ALLOCA_TP]])
    %else_joined_tp = stream.timepoint.join max(%parent_tp, %alloca_tp) => !stream.timepoint
    // CHECK:   %[[ELSE_TP:.+]] = stream.test.timeline_op await(%[[ELSE_JOINED]]) => with(%[[RESOURCE]]) : (!stream.resource<transient>{%[[SIZE]]})
    %else_tp = stream.test.timeline_op await(%else_joined_tp) =>
      with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    // CHECK:   scf.yield %[[ELSE_TP]]
    scf.yield %else_tp : !stream.timepoint
  }

  // Pass creates a final join combining alloca and if result.
  // CHECK: %{{.+}} = stream.timepoint.join max(%[[ALLOCA_TP]], %[[IF_RESULT]])
  // CHECK: %[[DEALLOCA:.+]] = stream.resource.dealloca origin await(%{{.+}}) => %[[RESOURCE]]
  // CHECK: util.return %[[DEALLOCA]]
  util.return %if_result : !stream.timepoint
}

// -----

// Tests that timepoint coverage correctly spans parent and nested scf.while regions.
// The while loop body joins a parent-scope timepoint with a loop-carried timepoint,
// requiring the coverage analysis to track timepoints across scope boundaries.

// CHECK-LABEL: @cross_scope_while_await_parent
// CHECK-SAME: (%[[PARENT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index, {{.+}}: index)
util.func private @cross_scope_while_await_parent(%parent_tp: !stream.timepoint, %size: index, %limit: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TP:.+]] = stream.resource.alloca
  %resource, %alloca_tp = stream.resource.alloca uninitialized await(%parent_tp) => !stream.resource<transient>{%size} => !stream.timepoint

  // While loop with parent timepoint in condition and body.
  // CHECK: %[[WHILE_RESULT:.+]]:2 = scf.while
  %while_result:2 = scf.while (%iter = %c0, %tp = %alloca_tp) : (index, !stream.timepoint) -> (index, !stream.timepoint) {
    %cond = arith.cmpi slt, %iter, %limit : index
    scf.condition(%cond) %iter, %tp : index, !stream.timepoint
  } do {
  ^bb0(%iter: index, %tp: !stream.timepoint):
    // Body joins parent_tp and loop-carried tp.
    // CHECK:   %[[JOINED:.+]] = stream.timepoint.join max(%[[PARENT_TP]], %{{.+}})
    %joined_tp = stream.timepoint.join max(%parent_tp, %tp) => !stream.timepoint
    // CHECK:   stream.test.timeline_op await(%[[JOINED]]) => with(%[[RESOURCE]]) : (!stream.resource<transient>{%[[SIZE]]})
    %cmd_tp = stream.test.timeline_op await(%joined_tp) =>
      with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    %next_iter = arith.addi %iter, %c1 : index
    scf.yield %next_iter, %cmd_tp : index, !stream.timepoint
  }

  // Pass creates a final join combining alloca and while result.
  // CHECK: %{{.+}} = stream.timepoint.join max(%[[ALLOCA_TP]], %[[WHILE_RESULT]]#1)
  // CHECK: %[[DEALLOCA:.+]] = stream.resource.dealloca origin await(%{{.+}}) => %[[RESOURCE]]
  // CHECK: util.return %[[DEALLOCA]]
  util.return %while_result#1 : !stream.timepoint
}

// -----

// Tests the conservative fallback when scf.for has NO timepoint result.
// When the SCF op doesn't yield a timepoint, we cannot track resource lifetimes
// through it, so captured resources are marked indeterminate (no deallocation).
// This tests the "return false" path in analyzeForLoop when getOrJoinTimepointResults
// returns nullopt.

// CHECK-LABEL: @for_no_timepoint_result_conservative
// CHECK-SAME: (%[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index)
util.func private @for_no_timepoint_result_conservative(%input_tp: !stream.timepoint, %size: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TP:.+]] = stream.resource.alloca uninitialized await(%[[INPUT_TP]]) => !stream.resource<transient>{%[[SIZE]]}
  %resource, %alloca_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint

  // This loop yields ONLY an index, no timepoint.
  // The pass cannot track resource lifetimes through this loop.
  // CHECK: %[[LOOP_RESULT:.+]] = scf.for
  %loop_result = scf.for %i = %c0 to %c10 step %c1 iter_args(%sum = %c0) -> index {
    // Use the captured resource inside the loop.
    // CHECK: stream.test.timeline_op await(%[[ALLOCA_TP]]) => with(%[[RESOURCE]]) : (!stream.resource<transient>{%[[SIZE]]})
    %cmd_tp = stream.test.timeline_op await(%alloca_tp) =>
      with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    %next_sum = arith.addi %sum, %c1 : index
    scf.yield %next_sum : index
  }

  // The captured resource should NOT have a deallocation inserted because
  // the pass could not analyze it (no timepoint result from loop).
  // CHECK-NOT: stream.resource.dealloca
  // CHECK: util.return %[[LOOP_RESULT]]
  util.return %loop_result : index
}

// -----

// Tests the conservative fallback when scf.if has NO timepoint result.
// Similar to the for loop case - when the if doesn't yield a timepoint,
// captured resources are marked indeterminate.

// CHECK-LABEL: @if_no_timepoint_result_conservative
// CHECK-SAME: ({{.+}}: i1, %[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index)
util.func private @if_no_timepoint_result_conservative(%cond: i1, %input_tp: !stream.timepoint, %size: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TP:.+]] = stream.resource.alloca uninitialized await(%[[INPUT_TP]]) => !stream.resource<transient>{%[[SIZE]]}
  %resource, %alloca_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint

  // This if yields ONLY an index, no timepoint.
  // CHECK: %[[IF_RESULT:.+]] = scf.if
  %if_result = scf.if %cond -> index {
    // CHECK: stream.test.timeline_op await(%[[ALLOCA_TP]]) => with(%[[RESOURCE]]) : (!stream.resource<transient>{%[[SIZE]]})
    %then_tp = stream.test.timeline_op await(%alloca_tp) =>
      with(%resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    scf.yield %c1 : index
  } else {
    scf.yield %c0 : index
  }

  // The captured resource should NOT have a deallocation inserted.
  // CHECK-NOT: stream.resource.dealloca
  // CHECK: util.return %[[IF_RESULT]]
  util.return %if_result : index
}

// -----

// Tests that a resource allocated inside a loop, yielded out, but NOT returned
// from the function is still properly deallocated. This is the "yielded then
// dropped" scenario.

// CHECK-LABEL: @loop_yielded_then_dropped
// CHECK-SAME: (%[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index, {{.+}}: !stream.resource<transient>)
util.func private @loop_yielded_then_dropped(%input_tp: !stream.timepoint, %size: index, %init_resource: !stream.resource<transient>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // CHECK: %[[LOOP_RESULT:.+]]:2 = scf.for
  %loop_resource, %loop_tp = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg_res = %init_resource, %arg_tp = %input_tp) -> (!stream.resource<transient>, !stream.timepoint) {
    // Allocate inside the loop (replacing the iter_arg resource each iteration).
    // CHECK: %[[LOCAL_RESOURCE:.+]], %[[LOCAL_ALLOCA_TP:.+]] = stream.resource.alloca
    %local_resource, %local_alloca_tp = stream.resource.alloca uninitialized await(%arg_tp) => !stream.resource<transient>{%size} => !stream.timepoint

    // CHECK: %[[CMD_TP:.+]] = stream.test.timeline_op await(%[[LOCAL_ALLOCA_TP]])
    %cmd_tp = stream.test.timeline_op await(%local_alloca_tp) =>
      with(%local_resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint

    // Resource is yielded out of the loop (new resource replaces arg_res).
    scf.yield %local_resource, %cmd_tp : !stream.resource<transient>, !stream.timepoint
  }

  // The yielded resource is NOT returned - it's dropped here.
  // The pass MUST insert a deallocation for %loop_resource after the loop.
  // CHECK: stream.resource.dealloca {{.*}}await(%[[LOOP_RESULT]]#1) => %[[LOOP_RESULT]]#0
  // CHECK: util.return
  util.return %loop_tp : !stream.timepoint
}

// -----

// Tests that a resource allocated inside an if-branch, yielded out, but NOT
// returned from the function is still properly deallocated.

// CHECK-LABEL: @if_yielded_then_dropped
// CHECK-SAME: ({{.+}}: i1, %[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index, {{.+}}: !stream.resource<transient>)
util.func private @if_yielded_then_dropped(%cond: i1, %input_tp: !stream.timepoint, %size: index, %else_resource: !stream.resource<transient>) -> !stream.timepoint {
  // CHECK: %[[IF_RESULT:.+]]:2 = scf.if
  %if_resource, %if_tp = scf.if %cond -> (!stream.resource<transient>, !stream.timepoint) {
    // Allocate inside the then-branch.
    // CHECK: %[[LOCAL_RESOURCE:.+]], %[[LOCAL_ALLOCA_TP:.+]] = stream.resource.alloca
    %local_resource, %local_alloca_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint

    // CHECK: %[[CMD_TP:.+]] = stream.test.timeline_op await(%[[LOCAL_ALLOCA_TP]])
    %cmd_tp = stream.test.timeline_op await(%local_alloca_tp) =>
      with(%local_resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint

    // Resource is yielded out of the if.
    scf.yield %local_resource, %cmd_tp : !stream.resource<transient>, !stream.timepoint
  } else {
    // Else-branch yields a different resource (defined outside, so indeterminate).
    scf.yield %else_resource, %input_tp : !stream.resource<transient>, !stream.timepoint
  }

  // The yielded resource is NOT returned - it's dropped here.
  // The pass MUST insert a deallocation for %if_resource after the if.
  // CHECK: stream.resource.dealloca {{.*}}await(%[[IF_RESULT]]#1) => %[[IF_RESULT]]#0
  // CHECK: util.return
  util.return %if_tp : !stream.timepoint
}
