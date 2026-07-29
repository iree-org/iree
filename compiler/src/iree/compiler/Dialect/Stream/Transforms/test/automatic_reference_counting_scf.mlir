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

// Tests that an alloca used as a resource iter_arg is not deallocated before
// the loop. The loop result may alias the initial resource when the loop has
// zero iterations, so ownership transfers to the loop result.

// CHECK-LABEL: @loop_iter_arg_initial_alloca_lifetime
// CHECK-SAME: (%[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index)
util.func private @loop_iter_arg_initial_alloca_lifetime(%input_tp: !stream.timepoint, %size: index) -> (!stream.resource<transient>, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // CHECK: %[[INIT_RESOURCE:.+]], %[[INIT_ALLOCA_TP:.+]] = stream.resource.alloca uninitialized await(%[[INPUT_TP]]) => !stream.resource<transient>{%[[SIZE]]}
  %init_resource, %init_alloca_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint

  // CHECK: %[[INIT_USE_TP:.+]] = stream.test.timeline_op await(%[[INIT_ALLOCA_TP]])
  %init_use_tp = stream.test.timeline_op await(%init_alloca_tp) =>
    with(%init_resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint

  // CHECK-NEXT: %[[LOOP_RESULT:.+]]:2 = scf.for
  %loop_resource, %loop_tp = scf.for %i = %c0 to %c10 step %c1 iter_args(%iter_resource = %init_resource, %iter_tp = %init_use_tp) -> (!stream.resource<transient>, !stream.timepoint) {
    %next_resource, %next_alloca_tp = stream.resource.alloca uninitialized await(%iter_tp) => !stream.resource<transient>{%size} => !stream.timepoint
    %cmd_tp = stream.test.timeline_op await(%iter_tp, %next_alloca_tp) =>
      with(%iter_resource, %next_resource) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
    scf.yield %next_resource, %cmd_tp : !stream.resource<transient>, !stream.timepoint
  }

  // CHECK: util.return %[[LOOP_RESULT]]#0, %[[LOOP_RESULT]]#1
  util.return %loop_resource, %loop_tp : !stream.resource<transient>, !stream.timepoint
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

// Tests that duplicate yielded resources are not deallocated through an unused
// sibling result. A single allocation may be yielded into multiple loop result
// positions and one result can escape while another is dropped.

// CHECK-LABEL: @loop_duplicate_yielded_resource
// CHECK-SAME: (%[[INPUT_TP:.+]]: !stream.timepoint, %[[SIZE:.+]]: index, {{.+}}: !stream.resource<transient>, {{.+}}: !stream.resource<transient>)
util.func private @loop_duplicate_yielded_resource(%input_tp: !stream.timepoint, %size: index, %init_a: !stream.resource<transient>, %init_b: !stream.resource<transient>) -> (!stream.resource<transient>, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  // CHECK: %[[LOOP_RESULT:.+]]:3 = scf.for
  %result_a, %result_b, %result_tp = scf.for %i = %c0 to %c10 step %c1 iter_args(%arg_a = %init_a, %arg_b = %init_b, %arg_tp = %input_tp) -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    // CHECK: %[[LOCAL_RESOURCE:.+]], %[[LOCAL_ALLOCA_TP:.+]] = stream.resource.alloca
    %local_resource, %local_alloca_tp = stream.resource.alloca uninitialized await(%arg_tp) => !stream.resource<transient>{%size} => !stream.timepoint

    // CHECK: %[[CMD_TP:.+]] = stream.test.timeline_op await(%[[LOCAL_ALLOCA_TP]])
    %cmd_tp = stream.test.timeline_op await(%local_alloca_tp) =>
      with(%local_resource) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint

    // CHECK: scf.yield %[[LOCAL_RESOURCE]], %[[LOCAL_RESOURCE]], %[[CMD_TP]]
    scf.yield %local_resource, %local_resource, %cmd_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }

  // CHECK-NOT: stream.resource.dealloca {{.*}}=> %[[LOOP_RESULT]]#1
  // CHECK: util.return %[[LOOP_RESULT]]#0, %[[LOOP_RESULT]]#2
  util.return %result_a, %result_tp : !stream.resource<transient>, !stream.timepoint
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

// -----

// Tests that ARC emits no deallocation for two control-flow results that denote
// the same buffer at runtime. A resource allocated inside an scf.if is yielded
// into two of that if's results, and the surrounding scf.for carries them into
// two loop results. On the true path both results are that local-scope allocation, on the
// fall-through path both are the same iter_arg. Deallocating each result
// would free the buffer twice and let the free race a still-live reader in 
// out-of-order executor.

// CHECK-LABEL: @sibling_aliased_control_flow_results
util.func private @sibling_aliased_control_flow_results(%cond: i1, %input_tp: !stream.timepoint, %size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %resource, %alloca_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: scf.for
  %loop:3 = scf.for %i = %c0 to %c4 step %c1 iter_args(%a0 = %resource, %a1 = %resource, %itp = %alloca_tp)
      -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    // CHECK: scf.if
    %if:2 = scf.if %cond -> (!stream.resource<transient>, !stream.resource<transient>) {
      %local, %ltp = stream.resource.alloca uninitialized await(%itp) => !stream.resource<transient>{%size} => !stream.timepoint
      %u = stream.test.timeline_op await(%ltp) =>
        with(%local) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
      // Both results are %local on this path, so %loop#0 and %loop#1 are one buffer.
      scf.yield %local, %local : !stream.resource<transient>, !stream.resource<transient>
    } else {
      scf.yield %a0, %a1 : !stream.resource<transient>, !stream.resource<transient>
    }
    scf.yield %if#0, %if#1, %itp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%loop#2) =>
    with(%loop#0, %loop#1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
  // The two results are one buffer, so neither may be deallocated (double-free).
  // CHECK-NOT: stream.resource.dealloca
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}

// -----

// Tests that ARC still detects aliasing when the two results are produced by
// *different* ops (not two results of a single op). Two independent scf.if ops
// each forward the same in-loop %shared allocation on their then-branch, so on
// any execution that enters those branches the two results denote the same
// buffer - even though nothing syntactically connects them (different ops,
// different SSA values, no tie). ARC cannot know at compile time which branch a
// conditional will take, so it must treat the two results as a possible alias
// and deallocate neither independently; doing so would double-free %shared on
// the executions where they coincide.

// CHECK-LABEL: @cross_op_aliased_loop_results
util.func private @cross_op_aliased_loop_results(%cond: i1, %input_tp: !stream.timepoint, %size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %init, %init_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: %[[LOOP:.+]]:3 = scf.for
  %loop:3 = scf.for %i = %c0 to %c4 step %c1 iter_args(%a0 = %init, %a1 = %init, %itp = %init_tp)
      -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    %shared, %stp = stream.resource.alloca uninitialized await(%itp) => !stream.resource<transient>{%size} => !stream.timepoint
    %u = stream.test.timeline_op await(%stp) =>
      with(%shared) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    // %if0 forwards %shared into loop result 0...
    %if0:2 = scf.if %cond -> (!stream.resource<transient>, !stream.timepoint) {
      scf.yield %shared, %u : !stream.resource<transient>, !stream.timepoint
    } else {
      scf.yield %a0, %itp : !stream.resource<transient>, !stream.timepoint
    }
    // ...and an unrelated %if1 forwards the same %shared into loop result 1.
    %if1:2 = scf.if %cond -> (!stream.resource<transient>, !stream.timepoint) {
      scf.yield %shared, %u : !stream.resource<transient>, !stream.timepoint
    } else {
      scf.yield %a1, %itp : !stream.resource<transient>, !stream.timepoint
    }
    %j = stream.timepoint.join max(%if0#1, %if1#1) => !stream.timepoint
    scf.yield %if0#0, %if1#0, %j : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%loop#2) =>
    with(%loop#0, %loop#1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
  // Both results are %shared, so neither may be deallocated (double-free).
  // CHECK-NOT: stream.resource.dealloca {{.*}}=> %[[LOOP]]#0
  // CHECK-NOT: stream.resource.dealloca {{.*}}=> %[[LOOP]]#1
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}

// -----

// Tests that ARC detects the same aliasing across an scf.while. The loop carries
// %shared in two loop-carried slots and scf.condition forwards both to the while
// results, so both results are that one buffer at runtime. ARC must not
// deallocate them independently.

// CHECK-LABEL: @while_aliased_results
util.func private @while_aliased_results(%input_tp: !stream.timepoint, %size: index, %bound: index, %shared: !stream.resource<transient>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[W:.+]]:4 = scf.while
  %w:4 = scf.while (%i = %c0, %r0 = %shared, %r1 = %shared, %tp = %input_tp) : (index, !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) -> (index, !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    %continue = arith.cmpi slt, %i, %bound : index
    scf.condition(%continue) %i, %r0, %r1, %tp : index, !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  } do {
  ^bb0(%bi: index, %b0: !stream.resource<transient>, %b1: !stream.resource<transient>, %btp: !stream.timepoint):
    %u = stream.test.timeline_op await(%btp) =>
      with(%b0) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    %ni = arith.addi %bi, %c1 : index
    scf.yield %ni, %b0, %b1, %u : index, !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%w#3) =>
    with(%w#1, %w#2) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
  // Both while results are %shared, so neither may be deallocated (double-free).
  // CHECK-NOT: stream.resource.dealloca {{.*}}=> %[[W]]#1
  // CHECK-NOT: stream.resource.dealloca {{.*}}=> %[[W]]#2
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}

// -----

// Tests that ARC still frees two results of one multi-result op when they are
// *distinct* buffers on every path (the precision counterpart to the aliasing
// cases). Each branch of the scf.if carries a separately-allocated, separately-
// initialized resource into each result, so the two results never denote the
// same buffer. ARC must deallocate each - being conservative about aliasing must
// not degrade into never freeing control-flow results, which would leak.

// CHECK-LABEL: @distinct_sibling_results_precise
util.func private @distinct_sibling_results_precise(%cond: i1, %input_tp: !stream.timepoint, %size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %init0, %tp_a = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
  %init1, %tp_b = stream.resource.alloca uninitialized await(%tp_a) => !stream.resource<transient>{%size} => !stream.timepoint
  %itp0 = stream.timepoint.join max(%tp_a, %tp_b) => !stream.timepoint
  // CHECK: %[[LOOP:.+]]:3 = scf.for
  %loop:3 = scf.for %i = %c0 to %c4 step %c1 iter_args(%a0 = %init0, %a1 = %init1, %itp = %itp0)
      -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    // Each result is a distinct buffer regardless of which branch runs: the then
    // branch yields two separately-allocated resources; the else branch forwards
    // two distinctly-initialized iter_args.
    %inner:3 = scf.if %cond -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
      %l0, %ltp0 = stream.resource.alloca uninitialized await(%itp) => !stream.resource<transient>{%size} => !stream.timepoint
      %l1, %ltp1 = stream.resource.alloca uninitialized await(%itp) => !stream.resource<transient>{%size} => !stream.timepoint
      %jj = stream.timepoint.join max(%ltp0, %ltp1) => !stream.timepoint
      %uu = stream.test.timeline_op await(%jj) =>
        with(%l0, %l1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
      scf.yield %l0, %l1, %uu : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    } else {
      scf.yield %a0, %a1, %itp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    }
    scf.yield %inner#0, %inner#1, %inner#2 : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%loop#2) =>
    with(%loop#0, %loop#1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
  // Distinct buffers, so each result must still be freed (not leaked).
  // CHECK-DAG: stream.resource.dealloca {{.*}}=> %[[LOOP]]#0
  // CHECK-DAG: stream.resource.dealloca {{.*}}=> %[[LOOP]]#1
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}
// -----

// Tests that a resource used only as an scf.while init is not deallocated before
// the loop that consumes it. The init is passed as an operand (not referenced
// inside either region) and is carried through the loop, then read after it via
// the while results; because the while results are the inits on a zero-trip, the
// init's lifetime extends to the results. Freeing it right after its alloca
// would leave the loop body and the post-loop reader using freed memory. This is
// the scf.while counterpart of @loop_iter_arg_initial_alloca_lifetime.

// CHECK-LABEL: @while_init_arg_lifetime
util.func private @while_init_arg_lifetime(%input_tp: !stream.timepoint, %size: index, %bound: index) -> (!stream.resource<transient>, !stream.timepoint) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT: stream.resource.dealloca
  %init, %init_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
  %w:2 = scf.while (%iter = %init, %tp = %init_tp) : (!stream.resource<transient>, !stream.timepoint) -> (!stream.resource<transient>, !stream.timepoint) {
    %cond = arith.cmpi slt, %c0, %bound : index
    scf.condition(%cond) %iter, %tp : !stream.resource<transient>, !stream.timepoint
  } do {
  ^bb0(%body_res: !stream.resource<transient>, %body_tp: !stream.timepoint):
    %cmd_tp = stream.test.timeline_op await(%body_tp) =>
      with(%body_res) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    scf.yield %body_res, %cmd_tp : !stream.resource<transient>, !stream.timepoint
  }
  // CHECK: util.return
  util.return %w#0, %w#1 : !stream.resource<transient>, !stream.timepoint
}

// -----

// Tests that ARC tracks aliasing across an scf.while when the values are
// reordered and the arities differ: 4 inits feed 3 results, and scf.condition
// swaps the two aliasing values (result 0 comes from init 1, result 1 from
// init 0). Both results still trace back to %shared, so ARC must not deallocate
// them independently - it cannot assume a result lines up with the same-indexed
// init, and must follow the actual operand routing.

// CHECK-LABEL: @while_asymmetric_reordered_aliased_results
util.func private @while_asymmetric_reordered_aliased_results(%input_tp: !stream.timepoint, %size: index, %bound: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK-NOT: stream.resource.dealloca
  %shared, %shared_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
  %other, %other_tp = stream.resource.alloca uninitialized await(%shared_tp) => !stream.resource<transient>{%size} => !stream.timepoint
  %w:3 = scf.while (%i0 = %shared, %i1 = %shared, %i2 = %other, %t = %other_tp)
      : (!stream.resource<transient>, !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint)
     -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    %cond = arith.cmpi slt, %c0, %bound : index
    // Reordered and narrowed: result 0 <- %i1, result 1 <- %i0, %i2 dropped.
    scf.condition(%cond) %i1, %i0, %t : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  } do {
  ^bb0(%a0: !stream.resource<transient>, %a1: !stream.resource<transient>, %at: !stream.timepoint):
    %cmd_tp = stream.test.timeline_op await(%at) =>
      with(%a0) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    scf.yield %a1, %a0, %other, %cmd_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%w#2) =>
    with(%w#0, %w#1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}

// -----

// Tests that ARC tracks aliasing transitively through three levels of nesting
// (scf.if inside scf.while inside scf.for). One allocation made in the innermost
// branch escapes into two distinct outer loop results, so the two results denote
// the same buffer only after that allocation is traced out through all three
// nested ops. ARC must still find the alias and not deallocate them independently.

// CHECK-LABEL: @deep_transitive_aliased_results
util.func private @deep_transitive_aliased_results(%cond: i1, %input_tp: !stream.timepoint, %size: index, %bound: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK-NOT: stream.resource.dealloca
  %init, %init_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
  %loop:3 = scf.for %i = %c0 to %c4 step %c1 iter_args(%f0 = %init, %f1 = %init, %ft = %init_tp)
      -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    %w:3 = scf.while (%w0 = %f0, %w1 = %f1, %wt = %ft)
        : (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint)
       -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
      %continue = arith.cmpi slt, %c0, %bound : index
      scf.condition(%continue) %w0, %w1, %wt : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    } do {
    ^bb0(%b0: !stream.resource<transient>, %b1: !stream.resource<transient>, %bt: !stream.timepoint):
      %if:3 = scf.if %cond -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
        %deep, %deep_tp = stream.resource.alloca uninitialized await(%bt) => !stream.resource<transient>{%size} => !stream.timepoint
        %deep_use = stream.test.timeline_op await(%deep_tp) =>
          with(%deep) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
        // One allocation into BOTH results, three levels down.
        scf.yield %deep, %deep, %deep_use : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
      } else {
        scf.yield %b0, %b1, %bt : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
      }
      scf.yield %if#0, %if#1, %if#2 : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    }
    scf.yield %w#0, %w#1, %w#2 : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%loop#2) =>
    with(%loop#0, %loop#1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}

// -----

// Tests that ARC unifies an alias established by a tied result before comparing
// loop results. One loop slot carries %local; the other carries the resource
// result of stream.timepoint.await on %local, which is tied to (the same
// allocation as) %local. The two loop results are distinct SSA values but one
// buffer, so ARC must not deallocate them independently.

// CHECK-LABEL: @tie_chain_aliased_across_loop_phi
util.func private @tie_chain_aliased_across_loop_phi(%cond: i1, %input_tp: !stream.timepoint, %size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK-NOT: stream.resource.dealloca
  %init, %init_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
  %loop:3 = scf.for %i = %c0 to %c4 step %c1 iter_args(%f0 = %init, %f1 = %init, %ft = %init_tp)
      -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    %if:3 = scf.if %cond -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
      %local, %local_tp = stream.resource.alloca uninitialized await(%ft) => !stream.resource<transient>{%size} => !stream.timepoint
      %use_tp = stream.test.timeline_op await(%local_tp) =>
        with(%local) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
      // Tied result: %awaited is the same allocation as %local.
      %awaited = stream.timepoint.await %use_tp => %local : !stream.resource<transient>{%size}
      scf.yield %local, %awaited, %use_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    } else {
      scf.yield %f0, %f1, %ft : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    }
    scf.yield %if#0, %if#1, %if#2 : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%loop#2) =>
    with(%loop#0, %loop#1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}

// -----

// Tests that ARC treats a stream.resource.subview as an alias of its source
// across a loop. One loop slot carries %local, the other a subview of %local. A
// subview is a view into its source allocation, so both loop results denote the
// same underlying buffer; freeing either would free that buffer, so ARC must not
// deallocate them independently.

// CHECK-LABEL: @subview_aliased_across_loop_phi
util.func private @subview_aliased_across_loop_phi(%cond: i1, %input_tp: !stream.timepoint, %size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  // CHECK-NOT: stream.resource.dealloca
  %init, %init_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
  %loop:3 = scf.for %i = %c0 to %c4 step %c1 iter_args(%f0 = %init, %f1 = %init, %ft = %init_tp)
      -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    %if:3 = scf.if %cond -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
      %local, %local_tp = stream.resource.alloca uninitialized await(%ft) => !stream.resource<transient>{%size} => !stream.timepoint
      %use_tp = stream.test.timeline_op await(%local_tp) =>
        with(%local) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
      %sub = stream.resource.subview %local[%c64] : !stream.resource<transient>{%size} -> !stream.resource<transient>{%c64}
      scf.yield %local, %sub, %use_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    } else {
      scf.yield %f0, %f1, %ft : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    }
    scf.yield %if#0, %if#1, %if#2 : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%loop#2) =>
    with(%loop#0, %loop#1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%c64}) -> () => !stream.timepoint
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}

// -----

// Tests that the double-free is avoided for a multi-region branch beyond the
// scf.for/if/while forms: an scf.index_switch yields one allocation into two of
// its results from a case region. The two results are that one buffer on the
// case path, so ARC must not deallocate them independently - whether it reasons
// about the switch precisely or conservatively leaves such results alone, the
// requirement is the same: no double-free.

// CHECK-LABEL: @index_switch_aliased_results
util.func private @index_switch_aliased_results(%flag: index, %input_tp: !stream.timepoint, %size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  // CHECK-NOT: stream.resource.dealloca
  %init, %init_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
  %loop:3 = scf.for %i = %c0 to %c4 step %c1 iter_args(%f0 = %init, %f1 = %init, %ft = %init_tp)
      -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    %sw:3 = scf.index_switch %flag -> !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    case 0 {
      %local, %local_tp = stream.resource.alloca uninitialized await(%ft) => !stream.resource<transient>{%size} => !stream.timepoint
      %use_tp = stream.test.timeline_op await(%local_tp) =>
        with(%local) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
      scf.yield %local, %local, %use_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    }
    default {
      scf.yield %f0, %f1, %ft : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    }
    scf.yield %sw#0, %sw#1, %sw#2 : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%loop#2) =>
    with(%loop#0, %loop#1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}

// -----

// Tests that ARC stays precise under deep nesting (the distinct-buffer
// counterpart of @deep_transitive_aliased_results). Two results are carried
// through scf.if inside scf.while inside scf.for, but their backings are distinct
// on every path (separate in-branch allocas, separate inits). ARC must still
// deallocate each - deep nesting must not make it give up and leave carried
// results unfreed.

// CHECK-LABEL: @deeply_nested_distinct_results_precise
util.func private @deeply_nested_distinct_results_precise(%cond: i1, %input_tp: !stream.timepoint, %size: index, %bound: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %init0, %tp_a = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
  %init1, %tp_b = stream.resource.alloca uninitialized await(%tp_a) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: %[[LOOP:.+]]:3 = scf.for
  %loop:3 = scf.for %i = %c0 to %c4 step %c1 iter_args(%f0 = %init0, %f1 = %init1, %ft = %tp_b)
      -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    %w:3 = scf.while (%w0 = %f0, %w1 = %f1, %wt = %ft)
        : (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint)
       -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
      %continue = arith.cmpi slt, %c0, %bound : index
      scf.condition(%continue) %w0, %w1, %wt : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    } do {
    ^bb0(%b0: !stream.resource<transient>, %b1: !stream.resource<transient>, %bt: !stream.timepoint):
      %if:3 = scf.if %cond -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
        %d0, %d0_tp = stream.resource.alloca uninitialized await(%bt) => !stream.resource<transient>{%size} => !stream.timepoint
        %d1, %d1_tp = stream.resource.alloca uninitialized await(%bt) => !stream.resource<transient>{%size} => !stream.timepoint
        %joined = stream.timepoint.join max(%d0_tp, %d1_tp) => !stream.timepoint
        %deep_use = stream.test.timeline_op await(%joined) =>
          with(%d0, %d1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
        scf.yield %d0, %d1, %deep_use : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
      } else {
        scf.yield %b0, %b1, %bt : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
      }
      scf.yield %if#0, %if#1, %if#2 : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    }
    scf.yield %w#0, %w#1, %w#2 : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%loop#2) =>
    with(%loop#0, %loop#1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
  // Distinct backings on every path => both results are still freed.
  // CHECK-DAG: stream.resource.dealloca {{.*}}=> %[[LOOP]]#0
  // CHECK-DAG: stream.resource.dealloca {{.*}}=> %[[LOOP]]#1
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}

// -----

// Tests that the double-free is avoided with no loop at all - the minimal shape
// is two nested scf.if ops. The inner scf.if yields the same %local into two of
// its results (where the alias is visible as the same SSA value); the outer
// scf.if forwards those two inner results - now distinct SSA values with no tie -
// into two of its own results. On any execution that enters both inner branches
// the two outer results are %local; ARC cannot know the branch outcomes, so it
// must treat them as a possible alias and not deallocate them independently. Two
// levels are what expose the defect: the outer op re-presents the inner alias as
// two different values that still denote one buffer.
//
// Paired with @nested_ifs_distinct_results_no_loop (same shape, genuinely
// distinct buffers, still freed) to show the merge here is driven by the
// aliasing rather than by ARC refusing to free control-flow results.

// CHECK-LABEL: @nested_ifs_aliased_results_no_loop
util.func private @nested_ifs_aliased_results_no_loop(%cond0: i1, %cond1: i1, %input_tp: !stream.timepoint, %size: index, %fallback0: !stream.resource<transient>, %fallback1: !stream.resource<transient>) -> !stream.timepoint {
  // CHECK-NOT: stream.resource.dealloca
  %outer:3 = scf.if %cond0 -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    %inner:3 = scf.if %cond1 -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
      %local, %local_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
      %use_tp = stream.test.timeline_op await(%local_tp) =>
        with(%local) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
      // Same %local into both inner results.
      scf.yield %local, %local, %use_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    } else {
      scf.yield %fallback0, %fallback1, %input_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    }
    // Forward the two inner results: distinct SSA values, same buffer on this path.
    scf.yield %inner#0, %inner#1, %inner#2 : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  } else {
    scf.yield %fallback0, %fallback1, %input_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%outer#2) =>
    with(%outer#0, %outer#1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}

// -----

// Tests that ARC still frees both results in the distinct-buffer control for
// @nested_ifs_aliased_results_no_loop: the same two-level nested scf.if shape,
// but the inner branch allocates two separate resources, so the two outer
// results never denote the same buffer. Both must still be deallocated.

// CHECK-LABEL: @nested_ifs_distinct_results_no_loop
util.func private @nested_ifs_distinct_results_no_loop(%cond0: i1, %cond1: i1, %input_tp: !stream.timepoint, %size: index, %fallback0: !stream.resource<transient>, %fallback1: !stream.resource<transient>) -> !stream.timepoint {
  // CHECK: %[[OUTER:.+]]:3 = scf.if
  %outer:3 = scf.if %cond0 -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    %inner:3 = scf.if %cond1 -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
      %local0, %local0_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
      %local1, %local1_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
      %joined = stream.timepoint.join max(%local0_tp, %local1_tp) => !stream.timepoint
      %use_tp = stream.test.timeline_op await(%joined) =>
        with(%local0, %local1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
      scf.yield %local0, %local1, %use_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    } else {
      scf.yield %fallback0, %fallback1, %input_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
    }
    scf.yield %inner#0, %inner#1, %inner#2 : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  } else {
    scf.yield %fallback0, %fallback1, %input_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%outer#2) =>
    with(%outer#0, %outer#1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
  // Distinct backings => both results are still freed.
  // CHECK-DAG: stream.resource.dealloca {{.*}}=> %[[OUTER]]#0
  // CHECK-DAG: stream.resource.dealloca {{.*}}=> %[[OUTER]]#1
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}

// -----

// Tests that ARC emits no deallocation when the same SSA resource is yielded
// into two results of a single scf.if - both results are that one buffer, so
// freeing each independently would double-free it.

// CHECK-LABEL: @if_same_resource_into_multiple_results
util.func private @if_same_resource_into_multiple_results(%cond: i1, %input_tp: !stream.timepoint, %size: index, %fallback0: !stream.resource<transient>, %fallback1: !stream.resource<transient>) -> !stream.timepoint {
  // CHECK-NOT: stream.resource.dealloca
  %if:3 = scf.if %cond -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    %local, %local_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
    %use_tp = stream.test.timeline_op await(%local_tp) =>
      with(%local) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    scf.yield %local, %local, %use_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  } else {
    scf.yield %fallback0, %fallback1, %input_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%if#2) =>
    with(%if#0, %if#1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}

// -----

// Tests that ARC still frees both results in the distinct-buffer counterpart of
// @if_same_resource_into_multiple_results: one scf.if yields two separately
// allocated resources into its two results, which never denote the same buffer.

// CHECK-LABEL: @if_distinct_resources_into_multiple_results
util.func private @if_distinct_resources_into_multiple_results(%cond: i1, %input_tp: !stream.timepoint, %size: index, %fallback0: !stream.resource<transient>, %fallback1: !stream.resource<transient>) -> !stream.timepoint {
  // CHECK: %[[IF:.+]]:3 = scf.if
  %if:3 = scf.if %cond -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    %local0, %local0_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
    %local1, %local1_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
    %joined = stream.timepoint.join max(%local0_tp, %local1_tp) => !stream.timepoint
    %use_tp = stream.test.timeline_op await(%joined) =>
      with(%local0, %local1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
    scf.yield %local0, %local1, %use_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  } else {
    scf.yield %fallback0, %fallback1, %input_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%if#2) =>
    with(%if#0, %if#1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
  // CHECK-DAG: stream.resource.dealloca {{.*}}=> %[[IF]]#0
  // CHECK-DAG: stream.resource.dealloca {{.*}}=> %[[IF]]#1
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}

// -----

// Tests that ARC recognizes a tied alias yielded alongside its source into two
// results of one scf.if: %local and the resource result of
// stream.timepoint.await on %local (tied to it, hence the same allocation) are
// distinct SSA values but one buffer, so neither may be deallocated independently.

// CHECK-LABEL: @if_tied_alias_into_multiple_results
util.func private @if_tied_alias_into_multiple_results(%cond: i1, %input_tp: !stream.timepoint, %size: index, %fallback0: !stream.resource<transient>, %fallback1: !stream.resource<transient>) -> !stream.timepoint {
  // CHECK-NOT: stream.resource.dealloca
  %if:3 = scf.if %cond -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    %local, %local_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
    %use_tp = stream.test.timeline_op await(%local_tp) =>
      with(%local) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    %awaited = stream.timepoint.await %use_tp => %local : !stream.resource<transient>{%size}
    scf.yield %local, %awaited, %use_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  } else {
    scf.yield %fallback0, %fallback1, %input_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%if#2) =>
    with(%if#0, %if#1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}

// -----

// Tests that ARC emits no deallocation when scf.condition forwards the same
// before-region argument into two while results - both are that one buffer, so
// freeing each independently would double-free it.

// CHECK-LABEL: @while_same_resource_into_multiple_results
util.func private @while_same_resource_into_multiple_results(%input_tp: !stream.timepoint, %size: index, %bound: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  // CHECK-NOT: stream.resource.dealloca
  %shared, %shared_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
  %w:3 = scf.while (%iter = %shared, %tp = %shared_tp) : (!stream.resource<transient>, !stream.timepoint) -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    %continue = arith.cmpi slt, %c0, %bound : index
    scf.condition(%continue) %iter, %iter, %tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  } do {
  ^bb0(%a0: !stream.resource<transient>, %a1: !stream.resource<transient>, %at: !stream.timepoint):
    %use_tp = stream.test.timeline_op await(%at) =>
      with(%a0) : (!stream.resource<transient>{%size}) -> () => !stream.timepoint
    scf.yield %a0, %use_tp : !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%w#2) =>
    with(%w#0, %w#1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}

// -----

// Tests that ARC still frees both results in the distinct-buffer counterpart of
// @while_same_resource_into_multiple_results: scf.condition forwards two
// distinct-init before-region arguments into two while results, which never
// denote the same buffer.

// CHECK-LABEL: @while_distinct_resources_into_multiple_results
util.func private @while_distinct_resources_into_multiple_results(%input_tp: !stream.timepoint, %size: index, %bound: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %res0, %res0_tp = stream.resource.alloca uninitialized await(%input_tp) => !stream.resource<transient>{%size} => !stream.timepoint
  %res1, %res1_tp = stream.resource.alloca uninitialized await(%res0_tp) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: %[[W:.+]]:3 = scf.while
  %w:3 = scf.while (%i0 = %res0, %i1 = %res1, %tp = %res1_tp) : (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) -> (!stream.resource<transient>, !stream.resource<transient>, !stream.timepoint) {
    %continue = arith.cmpi slt, %c0, %bound : index
    scf.condition(%continue) %i0, %i1, %tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  } do {
  ^bb0(%a0: !stream.resource<transient>, %a1: !stream.resource<transient>, %at: !stream.timepoint):
    %use_tp = stream.test.timeline_op await(%at) =>
      with(%a0, %a1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
    scf.yield %a0, %a1, %use_tp : !stream.resource<transient>, !stream.resource<transient>, !stream.timepoint
  }
  %fin = stream.test.timeline_op await(%w#2) =>
    with(%w#0, %w#1) : (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) -> () => !stream.timepoint
  // CHECK-DAG: stream.resource.dealloca {{.*}}=> %[[W]]#0
  // CHECK-DAG: stream.resource.dealloca {{.*}}=> %[[W]]#1
  // CHECK: util.return
  util.return %fin : !stream.timepoint
}
