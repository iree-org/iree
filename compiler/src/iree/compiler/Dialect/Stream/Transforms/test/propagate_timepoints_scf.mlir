// RUN: iree-opt --split-input-file --iree-stream-propagate-timepoints %s | FileCheck %s

// Tests that pipelined scf.for iterations await previous iteration timepoints.

// CHECK-LABEL: @scf_for_pipelined
util.func public @scf_for_pipelined(%arg: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index

  // Initial work produces timepoint.
  // CHECK: %[[INIT:.+]], %[[INIT_TP:.+]] = stream.test.timeline_op
  %init, %init_tp = stream.test.timeline_op with(%arg) : (!stream.resource<external>{%c128}) -> !stream.resource<external>{%c128} => !stream.timepoint

  // Await before loop.
  // CHECK: stream.timepoint.await %[[INIT_TP]] => %[[INIT]]
  %init_ready = stream.timepoint.await %init_tp => %init : !stream.resource<external>{%c128}

  // Loop: each iteration awaits previous iteration's work.
  // CHECK: %[[FOR_RESULT:.+]]:2 = scf.for %{{.+}} = %c0 to %c3 step %c1
  // CHECK-SAME: iter_args(%[[ITER:.+]] = %[[INIT]], %{{.+}} = %[[INIT_TP]])
  // CHECK-SAME: -> (!stream.resource<external>, !stream.timepoint)
  %result = scf.for %i = %c0 to %c3 step %c1 iter_args(%iter = %init_ready) -> !stream.resource<external> {
    // Process iter, produces new timepoint.
    // CHECK: %[[NEXT:.+]], %[[NEXT_TP:.+]] = stream.test.timeline_op with(%[[ITER]])
    %next, %next_tp = stream.test.timeline_op with(%iter) : (!stream.resource<external>{%c128}) -> !stream.resource<external>{%c128} => !stream.timepoint

    // Await for next iteration.
    // CHECK: stream.timepoint.await %[[NEXT_TP]] => %[[NEXT]]
    %next_ready = stream.timepoint.await %next_tp => %next : !stream.resource<external>{%c128}
    // CHECK: scf.yield %[[NEXT]], %[[NEXT_TP]] : !stream.resource<external>, !stream.timepoint
    scf.yield %next_ready : !stream.resource<external>
  }

  // CHECK: %[[SIZE:.+]] = stream.resource.size %[[FOR_RESULT]]#0
  // CHECK: %[[RESULT:.+]] = stream.timepoint.await %[[FOR_RESULT]]#1 => %[[FOR_RESULT]]#0 : !stream.resource<external>{%[[SIZE]]}
  // CHECK: util.return %[[RESULT]] : !stream.resource<external>
  util.return %result : !stream.resource<external>
}

// -----

// Tests that scf.for with independent parallel ops preserves concurrency.

// CHECK-LABEL: @scf_for_parallel_selective
util.func public @scf_for_parallel_selective(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>) -> (!stream.resource<external>, !stream.resource<external>) {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  // Two independent initial resources.
  // CHECK: %[[INIT_TP0:.+]] = stream.timepoint.immediate
  // CHECK: %[[INIT_TP1:.+]] = stream.timepoint.immediate
  // CHECK: %[[FOR_RESULT:.+]]:4 = scf.for %{{.+}} = %c0 to %c3 step %c1
  // CHECK-SAME: iter_args(%[[ITER0:.+]] = %arg0, %{{.+}} = %[[INIT_TP0]], %[[ITER1:.+]] = %arg1, %{{.+}} = %[[INIT_TP1]])
  // CHECK-SAME: -> (!stream.resource<external>, !stream.timepoint, !stream.resource<external>, !stream.timepoint)
  %result:2 = scf.for %i = %c0 to %c3 step %c1 iter_args(%iter0 = %arg0, %iter1 = %arg1) -> (!stream.resource<external>, !stream.resource<external>) {
    // Two INDEPENDENT operations (can run concurrently).
    // CHECK: %[[NEXT0:.+]], %[[NEXT_TP0:.+]] = stream.test.timeline_op with(%[[ITER0]])
    %next0, %tp0 = stream.test.timeline_op with(%iter0) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

    // CHECK: %[[NEXT1:.+]], %[[NEXT_TP1:.+]] = stream.test.timeline_op with(%[[ITER1]])
    %next1, %tp1 = stream.test.timeline_op with(%iter1) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

    // Await both before yielding.
    // CHECK: stream.timepoint.await %[[NEXT_TP0]] => %[[NEXT0]]
    %ready0 = stream.timepoint.await %tp0 => %next0 : !stream.resource<external>{%c64}
    // CHECK: stream.timepoint.await %[[NEXT_TP1]] => %[[NEXT1]]
    %ready1 = stream.timepoint.await %tp1 => %next1 : !stream.resource<external>{%c64}

    // CHECK: scf.yield %[[NEXT0]], %[[NEXT_TP0]], %[[NEXT1]], %[[NEXT_TP1]] : !stream.resource<external>, !stream.timepoint, !stream.resource<external>, !stream.timepoint
    scf.yield %ready0, %ready1 : !stream.resource<external>, !stream.resource<external>
  }

  // CHECK: %[[SIZE1:.+]] = stream.resource.size %[[FOR_RESULT]]#2
  // CHECK: %[[FINAL1:.+]] = stream.timepoint.await %[[FOR_RESULT]]#3 => %[[FOR_RESULT]]#2 : !stream.resource<external>{%[[SIZE1]]}
  // CHECK: %[[SIZE0:.+]] = stream.resource.size %[[FOR_RESULT]]#0
  // CHECK: %[[FINAL0:.+]] = stream.timepoint.await %[[FOR_RESULT]]#1 => %[[FOR_RESULT]]#0 : !stream.resource<external>{%[[SIZE0]]}
  // CHECK: util.return %[[FINAL0]], %[[FINAL1]] : !stream.resource<external>, !stream.resource<external>
  util.return %result#0, %result#1 : !stream.resource<external>, !stream.resource<external>
}

// -----

// Tests that scf.while correctly handles bidirectional timepoint flow.

// CHECK-LABEL: @scf_while_bidirectional
util.func public @scf_while_bidirectional(%arg: !stream.resource<external>, %limit: index) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index

  // Initial work.
  // CHECK: %[[INIT:.+]], %[[INIT_TP:.+]] = stream.test.timeline_op
  %init, %init_tp = stream.test.timeline_op with(%arg) : (!stream.resource<external>{%c128}) -> !stream.resource<external>{%c128} => !stream.timepoint

  // CHECK: stream.timepoint.await %[[INIT_TP]] => %[[INIT]]
  %init_ready = stream.timepoint.await %init_tp => %init : !stream.resource<external>{%c128}

  // While loop: condition needs to await body's timepoint.
  // CHECK: %[[WHILE_RESULT:.+]]:3 = scf.while
  // CHECK-SAME: (%[[BEFORE_ITER:.+]] = %[[INIT]], %[[BEFORE_TP:.+]] = %[[INIT_TP]], %[[BEFORE_COUNT:.+]] = %c0)
  // CHECK-SAME: : (!stream.resource<external>, !stream.timepoint, index)
  // CHECK-SAME: -> (!stream.resource<external>, !stream.timepoint, index)
  %result:2 = scf.while (%iter = %init_ready, %count = %c0) : (!stream.resource<external>, index) -> (!stream.resource<external>, index) {
    // Condition: check if should continue.
    // CHECK: %[[COND:.+]] = arith.cmpi slt, %[[BEFORE_COUNT]], %{{.+}}
    %cond = arith.cmpi slt, %count, %limit : index
    // CHECK: scf.condition(%[[COND]]) %[[BEFORE_ITER]], %[[BEFORE_TP]], %[[BEFORE_COUNT]] : !stream.resource<external>, !stream.timepoint, index
    scf.condition(%cond) %iter, %count : !stream.resource<external>, index
  } do {
  // CHECK: ^bb0(%[[BODY_ITER:.+]]: !stream.resource<external>, %{{.+}}: !stream.timepoint, %[[BODY_COUNT:.+]]: index):
  ^bb0(%body_iter: !stream.resource<external>, %body_count: index):
    // Body: produce next iteration's work.
    // CHECK: %[[NEXT:.+]], %[[NEXT_TP:.+]] = stream.test.timeline_op with(%[[BODY_ITER]])
    %next, %next_tp = stream.test.timeline_op with(%body_iter) : (!stream.resource<external>{%c128}) -> !stream.resource<external>{%c128} => !stream.timepoint

    // CHECK: stream.timepoint.await %[[NEXT_TP]] => %[[NEXT]]
    %next_ready = stream.timepoint.await %next_tp => %next : !stream.resource<external>{%c128}
    // CHECK: %[[NEXT_COUNT:.+]] = arith.addi %[[BODY_COUNT]], %c1
    %next_count = arith.addi %body_count, %c1 : index

    // CHECK: scf.yield %[[NEXT]], %[[NEXT_TP]], %[[NEXT_COUNT]] : !stream.resource<external>, !stream.timepoint, index
    scf.yield %next_ready, %next_count : !stream.resource<external>, index
  }

  // CHECK: %[[SIZE:.+]] = stream.resource.size %[[WHILE_RESULT]]#0
  // CHECK: %[[RESULT:.+]] = stream.timepoint.await %[[WHILE_RESULT]]#1 => %[[WHILE_RESULT]]#0 : !stream.resource<external>{%[[SIZE]]}
  // CHECK: util.return %[[RESULT]] : !stream.resource<external>
  util.return %result#0 : !stream.resource<external>
}

// -----

// Tests that scf.if correctly merges timepoints from both branches.

// CHECK-LABEL: @scf_if_merge
util.func public @scf_if_merge(%cond: i1, %arg0: !stream.resource<external>, %arg1: !stream.resource<external>) -> !stream.resource<external> {
  %c64 = arith.constant 64 : index

  // scf.if: both branches do async work.
  // CHECK: %[[IF_RESULT:.+]]:2 = scf.if %{{.+}} -> (!stream.resource<external>, !stream.timepoint)
  %result = scf.if %cond -> !stream.resource<external> {
    // Then: work on arg0.
    // CHECK: %[[THEN:.+]], %[[THEN_TP:.+]] = stream.test.timeline_op with(%arg1)
    %then_result, %then_tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

    // CHECK: stream.timepoint.await %[[THEN_TP]] => %[[THEN]]
    %then_ready = stream.timepoint.await %then_tp => %then_result : !stream.resource<external>{%c64}
    // CHECK: scf.yield %[[THEN]], %[[THEN_TP]] : !stream.resource<external>, !stream.timepoint
    scf.yield %then_ready : !stream.resource<external>
  } else {
    // Else: work on arg1.
    // CHECK: %[[ELSE:.+]], %[[ELSE_TP:.+]] = stream.test.timeline_op with(%arg2)
    %else_result, %else_tp = stream.test.timeline_op with(%arg1) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

    // CHECK: stream.timepoint.await %[[ELSE_TP]] => %[[ELSE]]
    %else_ready = stream.timepoint.await %else_tp => %else_result : !stream.resource<external>{%c64}
    // CHECK: scf.yield %[[ELSE]], %[[ELSE_TP]] : !stream.resource<external>, !stream.timepoint
    scf.yield %else_ready : !stream.resource<external>
  }

  // Use result (need to merge timepoints from both branches).
  // CHECK: %[[SIZE:.+]] = stream.resource.size %[[IF_RESULT]]#0
  // CHECK: %[[RESULT:.+]] = stream.timepoint.await %[[IF_RESULT]]#1 => %[[IF_RESULT]]#0 : !stream.resource<external>{%[[SIZE]]}
  // CHECK: %[[FINAL:.+]], %[[FINAL_TP:.+]] = stream.test.timeline_op with(%[[RESULT]])
  %final, %final_tp = stream.test.timeline_op with(%result) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

  // CHECK: %[[FINAL_READY:.+]] = stream.timepoint.await %[[FINAL_TP]] => %[[FINAL]]
  %final_ready = stream.timepoint.await %final_tp => %final : !stream.resource<external>{%c64}
  // CHECK: util.return %[[FINAL_READY]] : !stream.resource<external>
  util.return %final_ready : !stream.resource<external>
}

// -----

// Tests that timepoint mappings do not leak between scf.if branches.
// Regression test: if mappings leak, the else branch would try to use
// a timepoint defined in the then branch, causing a dominance error.

// CHECK-LABEL: @scf_if_no_sibling_leakage
// CHECK-SAME: %{{.+}}: i1, %[[ARG:.+]]: !stream.resource<external>
util.func public @scf_if_no_sibling_leakage(%cond: i1, %arg0: !stream.resource<external>) -> !stream.resource<external> {
  %c64 = arith.constant 64 : index

  // CHECK: %[[IF_RESULT:.+]]:2 = scf.if %{{.+}} -> (!stream.resource<external>, !stream.timepoint)
  %result = scf.if %cond -> !stream.resource<external> {
    // Then: Await the outer arg. This maps %arg0 -> %tp_then internally.
    // CHECK: %[[THEN_TP:.+]] = stream.timepoint.immediate
    %tp_then = stream.timepoint.immediate => !stream.timepoint
    // CHECK: stream.timepoint.await %[[THEN_TP]] => %[[ARG]]
    %r = stream.timepoint.await %tp_then => %arg0 : !stream.resource<external>{%c64}
    // CHECK: scf.yield %[[ARG]], %[[THEN_TP]] : !stream.resource<external>, !stream.timepoint
    scf.yield %r : !stream.resource<external>
  } else {
    // Else: Just pass through %arg0 unchanged. If sibling leakage occurred,
    // expandOperand would find %arg0 -> %tp_then and try to use %tp_then here,
    // which would be invalid since %tp_then is defined in the then branch.
    // CHECK: %[[ELSE_TP:.+]] = stream.timepoint.immediate
    // CHECK: scf.yield %[[ARG]], %[[ELSE_TP]] : !stream.resource<external>, !stream.timepoint
    scf.yield %arg0 : !stream.resource<external>
  }

  // CHECK: %[[SIZE:.+]] = stream.resource.size %[[IF_RESULT]]#0
  // CHECK: %[[RESULT:.+]] = stream.timepoint.await %[[IF_RESULT]]#1 => %[[IF_RESULT]]#0 : !stream.resource<external>{%[[SIZE]]}
  // CHECK: util.return %[[RESULT]] : !stream.resource<external>
  util.return %result : !stream.resource<external>
}

// -----

// Tests that nested scf.for inside scf.if correctly scopes timepoints.

// CHECK-LABEL: @nested_if_for
// CHECK-SAME: %{{.+}}: i1, %[[ARG:.+]]: !stream.resource<external>
util.func public @nested_if_for(%cond: i1, %arg: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  // CHECK: %[[IF_RESULT:.+]]:2 = scf.if %{{.+}} -> (!stream.resource<external>, !stream.timepoint)
  %result = scf.if %cond -> !stream.resource<external> {
    // Then: pipelined loop.
    // CHECK: %[[LOOP_INIT_TP:.+]] = stream.timepoint.immediate
    // CHECK: %[[LOOP_RESULT:.+]]:2 = scf.for %{{.+}} = %c0 to %c2 step %c1
    // CHECK-SAME: iter_args(%[[LOOP_ITER:.+]] = %[[ARG]], %{{.+}} = %[[LOOP_INIT_TP]])
    // CHECK-SAME: -> (!stream.resource<external>, !stream.timepoint)
    %loop_result = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter = %arg) -> !stream.resource<external> {
      // CHECK: %[[LOOP_NEXT:.+]], %[[LOOP_NEXT_TP:.+]] = stream.test.timeline_op with(%[[LOOP_ITER]])
      %next, %next_tp = stream.test.timeline_op with(%iter) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

      // CHECK: stream.timepoint.await %[[LOOP_NEXT_TP]] => %[[LOOP_NEXT]]
      %next_ready = stream.timepoint.await %next_tp => %next : !stream.resource<external>{%c64}
      // CHECK: scf.yield %[[LOOP_NEXT]], %[[LOOP_NEXT_TP]] : !stream.resource<external>, !stream.timepoint
      scf.yield %next_ready : !stream.resource<external>
    }
    // CHECK: %[[LOOP_SIZE:.+]] = stream.resource.size %[[LOOP_RESULT]]#0
    // CHECK: stream.timepoint.await %[[LOOP_RESULT]]#1 => %[[LOOP_RESULT]]#0 : !stream.resource<external>{%[[LOOP_SIZE]]}
    // CHECK: scf.yield %[[LOOP_RESULT]]#0, %[[LOOP_RESULT]]#1 : !stream.resource<external>, !stream.timepoint
    scf.yield %loop_result : !stream.resource<external>
  } else {
    // Else: single operation.
    // CHECK: %[[ELSE:.+]], %[[ELSE_TP:.+]] = stream.test.timeline_op with(%[[ARG]])
    %else_result, %else_tp = stream.test.timeline_op with(%arg) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

    // CHECK: stream.timepoint.await %[[ELSE_TP]] => %[[ELSE]]
    %else_ready = stream.timepoint.await %else_tp => %else_result : !stream.resource<external>{%c64}
    // CHECK: scf.yield %[[ELSE]], %[[ELSE_TP]] : !stream.resource<external>, !stream.timepoint
    scf.yield %else_ready : !stream.resource<external>
  }

  // CHECK: %[[FINAL_SIZE:.+]] = stream.resource.size %[[IF_RESULT]]#0
  // CHECK: %[[RESULT:.+]] = stream.timepoint.await %[[IF_RESULT]]#1 => %[[IF_RESULT]]#0 : !stream.resource<external>{%[[FINAL_SIZE]]}
  // CHECK: util.return %[[RESULT]] : !stream.resource<external>
  util.return %result : !stream.resource<external>
}

// -----

// Tests that multi-resource operations with selective joins preserve concurrency.

// CHECK-LABEL: @multi_resource_selective
util.func public @multi_resource_selective(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>) -> !stream.resource<external> {
  %c64 = arith.constant 64 : index

  // Two independent async operations.
  // CHECK: %[[RESULT0:.+]], %[[TP0:.+]] = stream.test.timeline_op with(%arg0)
  %result0, %tp0 = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

  // CHECK: %[[RESULT1:.+]], %[[TP1:.+]] = stream.test.timeline_op with(%arg1)
  %result1, %tp1 = stream.test.timeline_op with(%arg1) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

  // Await both.
  // CHECK: %[[READY0:.+]] = stream.timepoint.await %[[TP0]] => %[[RESULT0]]
  %ready0 = stream.timepoint.await %tp0 => %result0 : !stream.resource<external>{%c64}
  // CHECK: stream.timepoint.await %[[TP1]] => %[[RESULT1]]
  %ready1 = stream.timepoint.await %tp1 => %result1 : !stream.resource<external>{%c64}

  // Final operation uses ONLY result0 (should not wait for tp1).
  // CHECK: %[[FINAL:.+]], %[[FINAL_TP:.+]] = stream.test.timeline_op with(%[[READY0]])
  // CHECK-NOT: await(%[[TP1]])
  %final, %final_tp = stream.test.timeline_op with(%ready0) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

  // CHECK: %[[FINAL_READY:.+]] = stream.timepoint.await %[[FINAL_TP]] => %[[FINAL]]
  %final_ready = stream.timepoint.await %final_tp => %final : !stream.resource<external>{%c64}
  // CHECK: util.return %[[FINAL_READY]] : !stream.resource<external>
  util.return %final_ready : !stream.resource<external>
}

// -----

// Tests that pass does not create false dependencies between independent resources.

// CHECK-LABEL: @no_false_dependencies
util.func public @no_false_dependencies(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>) -> (!stream.resource<external>, !stream.resource<external>) {
  %c64 = arith.constant 64 : index

  // Timeline A: arg0 to result0.
  // CHECK: %[[RESULT0:.+]], %[[TP0:.+]] = stream.test.timeline_op with(%arg0)
  %result0, %tp0 = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

  // CHECK: %[[READY0:.+]] = stream.timepoint.await %[[TP0]] => %[[RESULT0]]
  %ready0 = stream.timepoint.await %tp0 => %result0 : !stream.resource<external>{%c64}

  // Timeline B: arg1 to result1 (completely independent).
  // CHECK: %[[RESULT1:.+]], %[[TP1:.+]] = stream.test.timeline_op with(%arg1)
  %result1, %tp1 = stream.test.timeline_op with(%arg1) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

  // CHECK: %[[READY1:.+]] = stream.timepoint.await %[[TP1]] => %[[RESULT1]]
  %ready1 = stream.timepoint.await %tp1 => %result1 : !stream.resource<external>{%c64}

  // Return both (should remain independent).
  // CHECK-NOT: stream.timepoint.join
  // CHECK: util.return %[[READY0]], %[[READY1]] : !stream.resource<external>, !stream.resource<external>
  util.return %ready0, %ready1 : !stream.resource<external>, !stream.resource<external>
}

// -----

// Tests that scf.index_switch correctly merges timepoints from all branches.

// CHECK-LABEL: @scf_index_switch_merge
util.func public @scf_index_switch_merge(%index: index, %arg0: !stream.resource<external>, %arg1: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  // scf.index_switch with multiple branches, each doing async work.
  // CHECK: %[[SWITCH_RESULT:.+]]:2 = scf.index_switch %{{.+}} -> !stream.resource<external>, !stream.timepoint
  %result = scf.index_switch %index -> !stream.resource<external>
  // CHECK: case 0 {
  case 0 {
    // Case 0: work on arg0.
    // CHECK: %[[CASE0:.+]], %[[CASE0_TP:.+]] = stream.test.timeline_op with(%arg1)
    %case0_res, %case0_tp = stream.test.timeline_op with(%arg0) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint
    // CHECK: stream.timepoint.await %[[CASE0_TP]] => %[[CASE0]]
    %case0_ready = stream.timepoint.await %case0_tp => %case0_res : !stream.resource<external>{%c64}
    // CHECK: scf.yield %[[CASE0]], %[[CASE0_TP]] : !stream.resource<external>, !stream.timepoint
    scf.yield %case0_ready : !stream.resource<external>
  }
  // CHECK: case 1 {
  case 1 {
    // Case 1: work on arg1.
    // CHECK: %[[CASE1:.+]], %[[CASE1_TP:.+]] = stream.test.timeline_op with(%arg2)
    %case1_res, %case1_tp = stream.test.timeline_op with(%arg1) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint
    // CHECK: stream.timepoint.await %[[CASE1_TP]] => %[[CASE1]]
    %case1_ready = stream.timepoint.await %case1_tp => %case1_res : !stream.resource<external>{%c64}
    // CHECK: scf.yield %[[CASE1]], %[[CASE1_TP]] : !stream.resource<external>, !stream.timepoint
    scf.yield %case1_ready : !stream.resource<external>
  }
  // CHECK: default {
  default {
    // Default: work on both.
    // CHECK: %[[DEFAULT:.+]], %[[DEFAULT_TP:.+]] = stream.test.timeline_op with(%arg1, %arg2)
    %default_res, %default_tp = stream.test.timeline_op with(%arg0, %arg1) : (!stream.resource<external>{%c64}, !stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint
    // CHECK: stream.timepoint.await %[[DEFAULT_TP]] => %[[DEFAULT]]
    %default_ready = stream.timepoint.await %default_tp => %default_res : !stream.resource<external>{%c64}
    // CHECK: scf.yield %[[DEFAULT]], %[[DEFAULT_TP]] : !stream.resource<external>, !stream.timepoint
    scf.yield %default_ready : !stream.resource<external>
  }

  // Final operation depends on merged timepoint from switch.
  // CHECK: %[[SIZE:.+]] = stream.resource.size %[[SWITCH_RESULT]]#0
  // CHECK: %[[MERGED:.+]] = stream.timepoint.await %[[SWITCH_RESULT]]#1 => %[[SWITCH_RESULT]]#0 : !stream.resource<external>{%[[SIZE]]}
  // CHECK: %[[FINAL:.+]], %[[FINAL_TP:.+]] = stream.test.timeline_op with(%[[MERGED]])
  %final, %final_tp = stream.test.timeline_op with(%result) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint
  // CHECK: %[[FINAL_READY:.+]] = stream.timepoint.await %[[FINAL_TP]] => %[[FINAL]]
  %final_ready = stream.timepoint.await %final_tp => %final : !stream.resource<external>{%c64}
  // CHECK: util.return %[[FINAL_READY]] : !stream.resource<external>
  util.return %final_ready : !stream.resource<external>
}

// -----

// Tests that scf.for with empty loop (upper ≤ lower) correctly propagates initial iter_args.

// CHECK-LABEL: @scf_for_empty_loop
util.func public @scf_for_empty_loop(%arg: !stream.resource<external>) -> !stream.resource<external> {
  %c5 = arith.constant 5 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index

  // Initial work produces timepoint.
  // CHECK: %[[INIT:.+]], %[[INIT_TP:.+]] = stream.test.timeline_op
  %init, %init_tp = stream.test.timeline_op with(%arg) : (!stream.resource<external>{%c128}) -> !stream.resource<external>{%c128} => !stream.timepoint

  // CHECK: stream.timepoint.await %[[INIT_TP]] => %[[INIT]]
  %init_ready = stream.timepoint.await %init_tp => %init : !stream.resource<external>{%c128}

  // Empty loop: 10 to 5 with step 1 executes 0 times.
  // CHECK: %[[FOR_RESULT:.+]]:2 = scf.for %{{.+}} = %c10 to %c5 step %c1
  // CHECK-SAME: iter_args(%[[ITER:.+]] = %[[INIT]], %{{.+}} = %[[INIT_TP]])
  // CHECK-SAME: -> (!stream.resource<external>, !stream.timepoint)
  %result = scf.for %i = %c10 to %c5 step %c1 iter_args(%iter = %init_ready) -> !stream.resource<external> {
    // Loop body: transformed but never executes at runtime.
    // CHECK: %[[NEXT:.+]], %[[NEXT_TP:.+]] = stream.test.timeline_op with(%[[ITER]])
    %next, %next_tp = stream.test.timeline_op with(%iter) : (!stream.resource<external>{%c128}) -> !stream.resource<external>{%c128} => !stream.timepoint
    // CHECK: stream.timepoint.await %[[NEXT_TP]] => %[[NEXT]]
    %next_ready = stream.timepoint.await %next_tp => %next : !stream.resource<external>{%c128}
    // CHECK: scf.yield %[[NEXT]], %[[NEXT_TP]] : !stream.resource<external>, !stream.timepoint
    scf.yield %next_ready : !stream.resource<external>
  }

  // Result equals initial values (loop never ran).
  // CHECK: %[[SIZE:.+]] = stream.resource.size %[[FOR_RESULT]]#0
  // CHECK: %[[RESULT:.+]] = stream.timepoint.await %[[FOR_RESULT]]#1 => %[[FOR_RESULT]]#0 : !stream.resource<external>{%[[SIZE]]}
  // CHECK: util.return %[[RESULT]] : !stream.resource<external>
  util.return %result : !stream.resource<external>
}

// -----

// Tests that scf.while with early exit correctly handles timepoint propagation.

// CHECK-LABEL: @scf_while_early_exit
util.func public @scf_while_early_exit(%arg: !stream.resource<external>, %threshold: index) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index

  // Initial work.
  // CHECK: %[[INIT:.+]], %[[INIT_TP:.+]] = stream.test.timeline_op
  %init, %init_tp = stream.test.timeline_op with(%arg) : (!stream.resource<external>{%c128}) -> !stream.resource<external>{%c128} => !stream.timepoint

  // CHECK: stream.timepoint.await %[[INIT_TP]] => %[[INIT]]
  %init_ready = stream.timepoint.await %init_tp => %init : !stream.resource<external>{%c128}

  // While loop with data-dependent early exit.
  // CHECK: %[[WHILE_RESULT:.+]]:2 = scf.while
  // CHECK-SAME: (%[[BEFORE_ITER:.+]] = %[[INIT]], %[[BEFORE_TP:.+]] = %[[INIT_TP]])
  // CHECK-SAME: : (!stream.resource<external>, !stream.timepoint)
  // CHECK-SAME: -> (!stream.resource<external>, !stream.timepoint)
  %result = scf.while (%iter = %init_ready) : (!stream.resource<external>) -> !stream.resource<external> {
    // Condition: query size and check against threshold for early exit.
    // CHECK: %[[ITER_SIZE:.+]] = stream.resource.size %[[BEFORE_ITER]]
    %iter_size = stream.resource.size %iter : !stream.resource<external>
    // CHECK: %[[COND:.+]] = arith.cmpi slt, %[[ITER_SIZE]], %{{.+}}
    %cond = arith.cmpi slt, %iter_size, %threshold : index
    // CHECK: scf.condition(%[[COND]]) %[[BEFORE_ITER]], %[[BEFORE_TP]] : !stream.resource<external>, !stream.timepoint
    scf.condition(%cond) %iter : !stream.resource<external>
  } do {
  // CHECK: ^bb0(%[[BODY_ITER:.+]]: !stream.resource<external>, %{{.+}}: !stream.timepoint):
  ^bb0(%body_iter: !stream.resource<external>):
    // Body: produce next iteration.
    // CHECK: %[[NEXT:.+]], %[[NEXT_TP:.+]] = stream.test.timeline_op with(%[[BODY_ITER]])
    %next, %next_tp = stream.test.timeline_op with(%body_iter) : (!stream.resource<external>{%c128}) -> !stream.resource<external>{%c128} => !stream.timepoint

    // CHECK: stream.timepoint.await %[[NEXT_TP]] => %[[NEXT]]
    %next_ready = stream.timepoint.await %next_tp => %next : !stream.resource<external>{%c128}

    // CHECK: scf.yield %[[NEXT]], %[[NEXT_TP]] : !stream.resource<external>, !stream.timepoint
    scf.yield %next_ready : !stream.resource<external>
  }

  // CHECK: %[[SIZE:.+]] = stream.resource.size %[[WHILE_RESULT]]#0
  // CHECK: %[[RESULT:.+]] = stream.timepoint.await %[[WHILE_RESULT]]#1 => %[[WHILE_RESULT]]#0 : !stream.resource<external>{%[[SIZE]]}
  // CHECK: util.return %[[RESULT]] : !stream.resource<external>
  util.return %result : !stream.resource<external>
}

// -----

// Tests that scf.while with timeline op IN the condition region correctly
// threads the timepoint from the op result (not the block arg timepoint).
// This tests the pass-by-reference fix for resourceTimepointMap.

// CHECK-LABEL: @scf_while_condition_with_timeline_op
util.func public @scf_while_condition_with_timeline_op(%arg: !stream.resource<external>, %limit: index) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c128 = arith.constant 128 : index

  // Initial work.
  // CHECK: %[[INIT:.+]], %[[INIT_TP:.+]] = stream.test.timeline_op
  %init, %init_tp = stream.test.timeline_op with(%arg) : (!stream.resource<external>{%c128}) -> !stream.resource<external>{%c128} => !stream.timepoint

  // CHECK: stream.timepoint.await %[[INIT_TP]] => %[[INIT]]
  %init_ready = stream.timepoint.await %init_tp => %init : !stream.resource<external>{%c128}

  // While loop: condition performs work on the resource and passes result.
  // CHECK: %[[WHILE_RESULT:.+]]:2 = scf.while
  // CHECK-SAME: (%[[BEFORE_ITER:.+]] = %[[INIT]], %[[BEFORE_TP:.+]] = %[[INIT_TP]])
  // CHECK-SAME: : (!stream.resource<external>, !stream.timepoint)
  // CHECK-SAME: -> (!stream.resource<external>, !stream.timepoint)
  %result = scf.while (%iter = %init_ready) : (!stream.resource<external>) -> !stream.resource<external> {
    // Condition: do work on the resource, then decide based on size.
    // CHECK: %[[COND_WORK:.+]], %[[COND_WORK_TP:.+]] = stream.test.timeline_op with(%[[BEFORE_ITER]])
    %cond_result, %cond_tp = stream.test.timeline_op with(%iter) : (!stream.resource<external>{%c128}) -> !stream.resource<external>{%c128} => !stream.timepoint

    // CHECK: %[[COND_READY:.+]] = stream.timepoint.await %[[COND_WORK_TP]] => %[[COND_WORK]]
    %cond_ready = stream.timepoint.await %cond_tp => %cond_result : !stream.resource<external>{%c128}

    // CHECK: %[[SIZE:.+]] = stream.resource.size %[[COND_READY]]
    %size = stream.resource.size %cond_ready : !stream.resource<external>
    // CHECK: %[[COND:.+]] = arith.cmpi slt, %[[SIZE]], %{{.+}}
    %cond = arith.cmpi slt, %size, %limit : index

    // The condition passes the MODIFIED resource with its NEW timepoint.
    // CHECK: scf.condition(%[[COND]]) %[[COND_WORK]], %[[COND_WORK_TP]] : !stream.resource<external>, !stream.timepoint
    scf.condition(%cond) %cond_ready : !stream.resource<external>
  } do {
  // CHECK: ^bb0(%[[BODY_ITER:.+]]: !stream.resource<external>, %[[BODY_TP:.+]]: !stream.timepoint):
  ^bb0(%body_iter: !stream.resource<external>):
    // Body: just pass through (no additional work) - threads the carried timepoint.
    // CHECK: scf.yield %[[BODY_ITER]], %[[BODY_TP]] : !stream.resource<external>, !stream.timepoint
    scf.yield %body_iter : !stream.resource<external>
  }

  // CHECK: %[[FINAL_SIZE:.+]] = stream.resource.size %[[WHILE_RESULT]]#0
  // CHECK: %[[FINAL:.+]] = stream.timepoint.await %[[WHILE_RESULT]]#1 => %[[WHILE_RESULT]]#0 : !stream.resource<external>{%[[FINAL_SIZE]]}
  // CHECK: util.return %[[FINAL]] : !stream.resource<external>
  util.return %result : !stream.resource<external>
}

// -----

// Tests RNN cell pattern with recurrent hidden state dependencies.

// CHECK-LABEL: @rnn_cell_recurrence
util.func public @rnn_cell_recurrence(%input: !stream.resource<external>, %initial_hidden: !stream.resource<external>, %seq_len: index) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  // RNN loop: hidden state recurs through iterations.
  // CHECK: %[[INIT_TP:.+]] = stream.timepoint.immediate
  // CHECK: %[[FOR_RESULT:.+]]:2 = scf.for %{{.+}} = %c0 to %{{.+}} step %c1
  // CHECK-SAME: iter_args(%[[HIDDEN:.+]] = %arg1, %{{.+}} = %[[INIT_TP]])
  // CHECK-SAME: -> (!stream.resource<external>, !stream.timepoint)
  %hidden_final = scf.for %t = %c0 to %seq_len step %c1 iter_args(%hidden = %initial_hidden) -> !stream.resource<external> {
    // RNN cell: combine input with previous hidden state.
    // CHECK: %[[CELL_OUT:.+]], %[[CELL_TP:.+]] = stream.test.timeline_op with(%arg0, %[[HIDDEN]])
    %cell_output, %cell_tp = stream.test.timeline_op with(%input, %hidden) : (!stream.resource<external>{%c64}, !stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

    // CHECK: stream.timepoint.await %[[CELL_TP]] => %[[CELL_OUT]]
    %next_hidden = stream.timepoint.await %cell_tp => %cell_output : !stream.resource<external>{%c64}

    // CHECK: scf.yield %[[CELL_OUT]], %[[CELL_TP]] : !stream.resource<external>, !stream.timepoint
    scf.yield %next_hidden : !stream.resource<external>
  }

  // CHECK: %[[SIZE:.+]] = stream.resource.size %[[FOR_RESULT]]#0
  // CHECK: %[[RESULT:.+]] = stream.timepoint.await %[[FOR_RESULT]]#1 => %[[FOR_RESULT]]#0 : !stream.resource<external>{%[[SIZE]]}
  // CHECK: util.return %[[RESULT]] : !stream.resource<external>
  util.return %hidden_final : !stream.resource<external>
}

// -----

// Tests that scf.for with read-only resource correctly handles timepoint propagation.

// CHECK-LABEL: @scf_for_readonly_resource
util.func public @scf_for_readonly_resource(%readonly: !stream.resource<external>, %initial: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  // Loop: readonly is passed through unchanged, initial is updated each iteration.
  // CHECK: %[[READONLY_TP:.+]] = stream.timepoint.immediate
  // CHECK: %[[INIT_TP:.+]] = stream.timepoint.immediate
  // CHECK: %[[FOR_RESULT:.+]]:4 = scf.for %{{.+}} = %c0 to %c3 step %c1
  // CHECK-SAME: iter_args(%[[RO:.+]] = %arg0, %[[RO_TP:.+]] = %[[READONLY_TP]], %[[ITER:.+]] = %arg1, %[[ITER_TP:.+]] = %[[INIT_TP]])
  // CHECK-SAME: -> (!stream.resource<external>, !stream.timepoint, !stream.resource<external>, !stream.timepoint)
  %result:2 = scf.for %i = %c0 to %c3 step %c1 iter_args(%ro = %readonly, %acc = %initial) -> (!stream.resource<external>, !stream.resource<external>) {
    // Use both readonly and accumulator resources.
    // CHECK: %[[NEXT:.+]], %[[NEXT_TP:.+]] = stream.test.timeline_op with(%[[RO]], %[[ITER]])
    %next, %next_tp = stream.test.timeline_op with(%ro, %acc) : (!stream.resource<external>{%c64}, !stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

    // CHECK: stream.timepoint.await %[[NEXT_TP]] => %[[NEXT]]
    %next_ready = stream.timepoint.await %next_tp => %next : !stream.resource<external>{%c64}

    // Readonly is passed through unchanged (with carried timepoint), accumulator is updated.
    // CHECK: scf.yield %[[RO]], %[[RO_TP]], %[[NEXT]], %[[NEXT_TP]] : !stream.resource<external>, !stream.timepoint, !stream.resource<external>, !stream.timepoint
    scf.yield %ro, %next_ready : !stream.resource<external>, !stream.resource<external>
  }

  // Return accumulator result (readonly is discarded).
  // CHECK: %[[SIZE:.+]] = stream.resource.size %[[FOR_RESULT]]#2
  // CHECK: %[[RESULT:.+]] = stream.timepoint.await %[[FOR_RESULT]]#3 => %[[FOR_RESULT]]#2 : !stream.resource<external>{%[[SIZE]]}
  // CHECK: util.return %[[RESULT]] : !stream.resource<external>
  util.return %result#1 : !stream.resource<external>
}

// -----

// Tests that partial chained dependencies don't create unnecessary joins.

// CHECK-LABEL: @partial_chained_dependencies
util.func public @partial_chained_dependencies(%arg: !stream.resource<external>) -> !stream.resource<external> {
  %c64 = arith.constant 64 : index

  // A: initial work.
  // CHECK: %[[A:.+]], %[[A_TP:.+]] = stream.test.timeline_op with(%arg0)
  %a, %a_tp = stream.test.timeline_op with(%arg) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

  // CHECK: %[[A_READY:.+]] = stream.timepoint.await %[[A_TP]] => %[[A]]
  %a_ready = stream.timepoint.await %a_tp => %a : !stream.resource<external>{%c64}

  // B: depends on A.
  // CHECK: %[[B:.+]], %[[B_TP:.+]] = stream.test.timeline_op with(%[[A_READY]])
  %b, %b_tp = stream.test.timeline_op with(%a_ready) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

  // CHECK: %[[B_READY:.+]] = stream.timepoint.await %[[B_TP]] => %[[B]]
  %b_ready = stream.timepoint.await %b_tp => %b : !stream.resource<external>{%c64}

  // C: depends on B only, NOT on A.
  // CHECK: %[[C:.+]], %[[C_TP:.+]] = stream.test.timeline_op with(%[[B_READY]])
  // CHECK-NOT: with(%[[A_READY]]
  %c, %c_tp = stream.test.timeline_op with(%b_ready) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

  // CHECK: %[[C_READY:.+]] = stream.timepoint.await %[[C_TP]] => %[[C]]
  %c_ready = stream.timepoint.await %c_tp => %c : !stream.resource<external>{%c64}

  // CHECK: util.return %[[C_READY]] : !stream.resource<external>
  util.return %c_ready : !stream.resource<external>
}

// -----

// Tests beam search pattern with multiple independent candidates.

// CHECK-LABEL: @beam_search_pattern
util.func public @beam_search_pattern(%beam0: !stream.resource<external>, %beam1: !stream.resource<external>, %seq_len: index) -> (!stream.resource<external>, !stream.resource<external>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  // Beam search loop: maintain 2 independent beams.
  // CHECK: %[[BEAM0_TP:.+]] = stream.timepoint.immediate
  // CHECK: %[[BEAM1_TP:.+]] = stream.timepoint.immediate
  // CHECK: %[[FOR_RESULT:.+]]:4 = scf.for %{{.+}} = %c0 to %{{.+}} step %c1
  // CHECK-SAME: iter_args(%[[B0:.+]] = %arg0, %{{.+}} = %[[BEAM0_TP]], %[[B1:.+]] = %arg1, %{{.+}} = %[[BEAM1_TP]])
  // CHECK-SAME: -> (!stream.resource<external>, !stream.timepoint, !stream.resource<external>, !stream.timepoint)
  %beams:2 = scf.for %t = %c0 to %seq_len step %c1 iter_args(%b0 = %beam0, %b1 = %beam1) -> (!stream.resource<external>, !stream.resource<external>) {
    // Update beam0 independently.
    // CHECK: %[[NEXT_B0:.+]], %[[NEXT_B0_TP:.+]] = stream.test.timeline_op with(%[[B0]])
    %next_b0, %tp0 = stream.test.timeline_op with(%b0) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

    // Update beam1 independently (no dependency on beam0).
    // CHECK: %[[NEXT_B1:.+]], %[[NEXT_B1_TP:.+]] = stream.test.timeline_op with(%[[B1]])
    %next_b1, %tp1 = stream.test.timeline_op with(%b1) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

    // Await both beams.
    // CHECK: stream.timepoint.await %[[NEXT_B0_TP]] => %[[NEXT_B0]]
    %ready_b0 = stream.timepoint.await %tp0 => %next_b0 : !stream.resource<external>{%c64}
    // CHECK: stream.timepoint.await %[[NEXT_B1_TP]] => %[[NEXT_B1]]
    %ready_b1 = stream.timepoint.await %tp1 => %next_b1 : !stream.resource<external>{%c64}

    // Yield both beams.
    // CHECK: scf.yield %[[NEXT_B0]], %[[NEXT_B0_TP]], %[[NEXT_B1]], %[[NEXT_B1_TP]] : !stream.resource<external>, !stream.timepoint, !stream.resource<external>, !stream.timepoint
    scf.yield %ready_b0, %ready_b1 : !stream.resource<external>, !stream.resource<external>
  }

  // Return both final beams (processed in reverse order).
  // CHECK: %[[SIZE1:.+]] = stream.resource.size %[[FOR_RESULT]]#2
  // CHECK: %[[FINAL_B1:.+]] = stream.timepoint.await %[[FOR_RESULT]]#3 => %[[FOR_RESULT]]#2 : !stream.resource<external>{%[[SIZE1]]}
  // CHECK: %[[SIZE0:.+]] = stream.resource.size %[[FOR_RESULT]]#0
  // CHECK: %[[FINAL_B0:.+]] = stream.timepoint.await %[[FOR_RESULT]]#1 => %[[FOR_RESULT]]#0 : !stream.resource<external>{%[[SIZE0]]}
  // CHECK: util.return %[[FINAL_B0]], %[[FINAL_B1]] : !stream.resource<external>, !stream.resource<external>
  util.return %beams#0, %beams#1 : !stream.resource<external>, !stream.resource<external>
}

// -----

// Tests pipeline parallelism with multiple stages running concurrently.

// CHECK-LABEL: @pipeline_parallelism
util.func public @pipeline_parallelism(%cond: i1, %data: !stream.resource<external>) -> !stream.resource<external> {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index

  // Pipeline loop: alternate between two stages based on condition.
  // CHECK: %[[INIT_TP:.+]] = stream.timepoint.immediate
  // CHECK: %[[FOR_RESULT:.+]]:2 = scf.for %{{.+}} = %c0 to %c2 step %c1
  // CHECK-SAME: iter_args(%[[ITER:.+]] = %arg1, %{{.+}} = %[[INIT_TP]])
  // CHECK-SAME: -> (!stream.resource<external>, !stream.timepoint)
  %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %data) -> !stream.resource<external> {
    // Pipeline stage selection.
    // CHECK: %[[STAGE_IF:.+]]:2 = scf.if {{.+}} -> (!stream.resource<external>, !stream.timepoint)
    %stage_result = scf.if %cond -> !stream.resource<external> {
      // Stage 1: preprocessing.
      // CHECK: %[[STAGE1:.+]], %[[STAGE1_TP:.+]] = stream.test.timeline_op with(%[[ITER]])
      %s1, %s1_tp = stream.test.timeline_op with(%acc) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

      // CHECK: stream.timepoint.await %[[STAGE1_TP]] => %[[STAGE1]]
      %s1_ready = stream.timepoint.await %s1_tp => %s1 : !stream.resource<external>{%c64}

      // CHECK: scf.yield %[[STAGE1]], %[[STAGE1_TP]] : !stream.resource<external>, !stream.timepoint
      scf.yield %s1_ready : !stream.resource<external>
    } else {
      // Stage 2: processing.
      // CHECK: %[[STAGE2:.+]], %[[STAGE2_TP:.+]] = stream.test.timeline_op with(%[[ITER]])
      %s2, %s2_tp = stream.test.timeline_op with(%acc) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint

      // CHECK: stream.timepoint.await %[[STAGE2_TP]] => %[[STAGE2]]
      %s2_ready = stream.timepoint.await %s2_tp => %s2 : !stream.resource<external>{%c64}

      // CHECK: scf.yield %[[STAGE2]], %[[STAGE2_TP]] : !stream.resource<external>, !stream.timepoint
      scf.yield %s2_ready : !stream.resource<external>
    }

    // CHECK: %[[STAGE_SIZE:.+]] = stream.resource.size %[[STAGE_IF]]#0
    // CHECK: stream.timepoint.await %[[STAGE_IF]]#1 => %[[STAGE_IF]]#0 : !stream.resource<external>{%[[STAGE_SIZE]]}
    // CHECK: scf.yield %[[STAGE_IF]]#0, %[[STAGE_IF]]#1 : !stream.resource<external>, !stream.timepoint
    scf.yield %stage_result : !stream.resource<external>
  }

  // CHECK: %[[SIZE:.+]] = stream.resource.size %[[FOR_RESULT]]#0
  // CHECK: %[[FINAL:.+]] = stream.timepoint.await %[[FOR_RESULT]]#1 => %[[FOR_RESULT]]#0 : !stream.resource<external>{%[[SIZE]]}
  // CHECK: util.return %[[FINAL]] : !stream.resource<external>
  util.return %result : !stream.resource<external>
}

// -----

// Deeply nested control flow stress test exercising:
// 1. Pass-by-reference: timeline op in while condition must thread timepoint
// 2. Sibling isolation: scf.if branches must not share timepoint mappings
// 3. Check-after-add: awaits must be inserted for outer-scope resources
// 4. Multi-level nesting: scf.for → scf.if → scf.while with correct flow
// 5. Cross-level resource flow: outer resources used/modified at inner levels

// CHECK-LABEL: @deeply_nested_stress_test
// CHECK-SAME: %[[ARG0:.+]]: !stream.resource<external>, %[[ARG1:.+]]: !stream.resource<external>
util.func public @deeply_nested_stress_test(
    %arg0: !stream.resource<external>,
    %arg1: !stream.resource<external>,
    %outer_cond: i1,
    %limit: index) -> (!stream.resource<external>, !stream.resource<external>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c64 = arith.constant 64 : index

  // Outer scf.for: carries two independent resources through iterations.
  // CHECK: %[[ARG0_TP:.+]] = stream.timepoint.immediate
  // CHECK: %[[ARG1_TP:.+]] = stream.timepoint.immediate
  // CHECK: %[[FOR:.+]]:4 = scf.for %[[I:.+]] = %c0 to %c2 step %c1
  // CHECK-SAME: iter_args(%[[FOR_RES0:.+]] = %[[ARG0]], %[[FOR_TP0:.+]] = %[[ARG0_TP]], %[[FOR_RES1:.+]] = %[[ARG1]], %[[FOR_TP1:.+]] = %[[ARG1_TP]])
  %result:2 = scf.for %i = %c0 to %c2 step %c1
      iter_args(%iter0 = %arg0, %iter1 = %arg1) -> (!stream.resource<external>, !stream.resource<external>) {

    // Inner scf.if: branches do DIFFERENT async work on DIFFERENT resources.
    // This tests sibling isolation - then branch's mappings must not leak to else.
    // CHECK: %[[IF:.+]]:4 = scf.if %{{.+}} -> (!stream.resource<external>, !stream.timepoint, !stream.resource<external>, !stream.timepoint)
    %if_result:2 = scf.if %outer_cond -> (!stream.resource<external>, !stream.resource<external>) {
      // THEN branch: contains scf.while with timeline op IN condition.
      // This tests pass-by-reference - condition's timeline op mapping must persist.

      // CHECK: %[[WHILE:.+]]:2 = scf.while (%[[W_RES:.+]] = %[[FOR_RES0]], %[[W_TP:.+]] = %[[FOR_TP0]])
      %while_result = scf.while (%w_iter = %iter0) : (!stream.resource<external>) -> !stream.resource<external> {
        // Condition: timeline op operates on iter, result passed to scf.condition.
        // CHECK: %[[COND_WORK:.+]], %[[COND_WORK_TP:.+]] = stream.test.timeline_op with(%[[W_RES]])
        %cond_res, %cond_tp = stream.test.timeline_op with(%w_iter) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint
        // CHECK: stream.timepoint.await %[[COND_WORK_TP]] => %[[COND_WORK]]
        %cond_ready = stream.timepoint.await %cond_tp => %cond_res : !stream.resource<external>{%c64}
        %size = stream.resource.size %cond_ready : !stream.resource<external>
        %cond = arith.cmpi slt, %size, %limit : index
        // CHECK: scf.condition(%{{.+}}) %[[COND_WORK]], %[[COND_WORK_TP]] : !stream.resource<external>, !stream.timepoint
        scf.condition(%cond) %cond_ready : !stream.resource<external>
      } do {
      ^bb0(%body_iter: !stream.resource<external>):
        // Body: pass through (tests block arg timepoint threading).
        // Capture block args and verify they're yielded back unchanged.
        // CHECK: ^bb0(%[[BODY_RES:.+]]: !stream.resource<external>, %[[BODY_TP:.+]]: !stream.timepoint):
        // CHECK:   scf.yield %[[BODY_RES]], %[[BODY_TP]] : !stream.resource<external>, !stream.timepoint
        scf.yield %body_iter : !stream.resource<external>
      }

      // Then branch yields while result AND the OTHER resource (iter1) unchanged.
      // iter1 keeps its timepoint from the for loop - no new immediate needed.
      // CHECK: %[[THEN_SIZE:.+]] = stream.resource.size %[[WHILE]]#0
      // CHECK: stream.timepoint.await %[[WHILE]]#1 => %[[WHILE]]#0 : !stream.resource<external>{%[[THEN_SIZE]]}
      // CHECK: scf.yield %[[WHILE]]#0, %[[WHILE]]#1, %[[FOR_RES1]], %[[FOR_TP1]]
      scf.yield %while_result, %iter1 : !stream.resource<external>, !stream.resource<external>
    } else {
      // ELSE branch: simple timeline ops on BOTH resources.
      // Tests that else branch doesn't see then branch's while mappings.

      // CHECK: %[[ELSE0:.+]], %[[ELSE0_TP:.+]] = stream.test.timeline_op with(%[[FOR_RES0]])
      %else0, %else0_tp = stream.test.timeline_op with(%iter0) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint
      // CHECK: stream.timepoint.await %[[ELSE0_TP]] => %[[ELSE0]]
      %else0_ready = stream.timepoint.await %else0_tp => %else0 : !stream.resource<external>{%c64}

      // CHECK: %[[ELSE1:.+]], %[[ELSE1_TP:.+]] = stream.test.timeline_op with(%[[FOR_RES1]])
      %else1, %else1_tp = stream.test.timeline_op with(%iter1) : (!stream.resource<external>{%c64}) -> !stream.resource<external>{%c64} => !stream.timepoint
      // CHECK: stream.timepoint.await %[[ELSE1_TP]] => %[[ELSE1]]
      %else1_ready = stream.timepoint.await %else1_tp => %else1 : !stream.resource<external>{%c64}

      // CHECK: scf.yield %[[ELSE0]], %[[ELSE0_TP]], %[[ELSE1]], %[[ELSE1_TP]]
      scf.yield %else0_ready, %else1_ready : !stream.resource<external>, !stream.resource<external>
    }

    // Awaits after scf.if (these sink down from original code).
    // CHECK: %[[IF_SIZE1:.+]] = stream.resource.size %[[IF]]#2
    // CHECK: stream.timepoint.await %[[IF]]#3 => %[[IF]]#2 : !stream.resource<external>{%[[IF_SIZE1]]}
    // CHECK: %[[IF_SIZE0:.+]] = stream.resource.size %[[IF]]#0
    // CHECK: stream.timepoint.await %[[IF]]#1 => %[[IF]]#0 : !stream.resource<external>{%[[IF_SIZE0]]}
    // For yield: pass if results back up.
    // CHECK: scf.yield %[[IF]]#0, %[[IF]]#1, %[[IF]]#2, %[[IF]]#3
    scf.yield %if_result#0, %if_result#1 : !stream.resource<external>, !stream.resource<external>
  }

  // Final awaits and return.
  // CHECK: %[[FINAL_SIZE1:.+]] = stream.resource.size %[[FOR]]#2
  // CHECK: %[[FINAL1:.+]] = stream.timepoint.await %[[FOR]]#3 => %[[FOR]]#2 : !stream.resource<external>{%[[FINAL_SIZE1]]}
  // CHECK: %[[FINAL_SIZE0:.+]] = stream.resource.size %[[FOR]]#0
  // CHECK: %[[FINAL0:.+]] = stream.timepoint.await %[[FOR]]#1 => %[[FOR]]#0 : !stream.resource<external>{%[[FINAL_SIZE0]]}
  // CHECK: util.return %[[FINAL0]], %[[FINAL1]]
  util.return %result#0, %result#1 : !stream.resource<external>, !stream.resource<external>
}
