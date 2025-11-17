// RUN: iree-opt --split-input-file --iree-stream-emplace-transients --verify-diagnostics %s | FileCheck %s

// Tests SCF control flow with transient in one branch.

// CHECK-LABEL: @scf_if_transient_in_then
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>, %[[STORAGE_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[COND:[a-z0-9]+]]: i1,
// CHECK-SAME:  %[[INPUT_TIMEPOINT:[a-z0-9]+]]: !stream.timepoint)
util.func public @scf_if_transient_in_then(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index,
  %cond: i1,
  %input_timepoint: !stream.timepoint
) -> (!stream.resource<*>, index) {
  // CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32

  // Timeline flows through scf.if - transient allocated in then branch awaiting input timepoint.
  // CHECK: %[[FINAL_TP:.+]] = scf.if %[[COND]] -> (!stream.timepoint) {
  %final_tp = scf.if %cond -> !stream.timepoint {
    // CHECK: %[[TRANSIENT_SIZE:.+]] = arith.constant 512 : index
    %c512 = arith.constant 512 : index
    // Size computation clones the constant to hoist it for pack computation.
    // CHECK: %[[TRANSIENT_SIZE_CONSERVATIVE:.+]] = arith.constant 512 : index
    // Pack with single slice (only then branch has allocation) - inserted inside scf.if since alloca dominance is local to this region.
    // Pack uses the conservative (cloned) size.
    // CHECK: %[[PACK:.+]]:2 = stream.resource.pack slices({
    // CHECK-NEXT:   [0, 0] = %[[TRANSIENT_SIZE_CONSERVATIVE]]
    // CHECK-NEXT: }) : index attributes {stream.experimental.transients}
    // Storage subview at offset 0.
    // CHECK: %[[BASE_OFFSET:.+]] = arith.constant 0 : index
    // CHECK: %[[STORAGE_VIEW:.+]] = stream.resource.subview %[[STORAGE]][%[[BASE_OFFSET]]] : !stream.resource<transient>{%[[STORAGE_SIZE]]} -> !stream.resource<transient>{%[[PACK]]#0}
    // Transient subview uses the original alloca size (not conservative).
    // CHECK: %[[TRANSIENT_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_VIEW]][%[[PACK]]#1]
    // CHECK-SAME: !stream.resource<transient>{%[[PACK]]#0} -> !stream.resource<transient>{%[[TRANSIENT_SIZE]]}
    %t0, %tp0 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c512} => !stream.timepoint
    // CRITICAL: Execute awaits input timepoint (alloca removed, timeline preserved).
    // CHECK: %[[EXEC_TP:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[TRANSIENT_SUBVIEW]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[TRANSIENT_SIZE]]}) {
    // CHECK:   stream.cmd.fill %[[ZERO_I32]], %[[CAPTURE]][%[[ZERO_INDEX]] for %[[TRANSIENT_SIZE]]] : i32 -> !stream.resource<transient>{%[[TRANSIENT_SIZE]]}
    // CHECK: } => !stream.timepoint
    %e0 = stream.cmd.execute await(%tp0) => with(%t0 as %c: !stream.resource<transient>{%c512}) {
      stream.cmd.fill %c0_i32, %c[%c0 for %c512] : i32 -> !stream.resource<transient>{%c512}
    } => !stream.timepoint
    %dt0 = stream.resource.dealloca await(%e0) => %t0 : !stream.resource<transient>{%c512} => !stream.timepoint
    // CHECK: scf.yield %[[EXEC_TP]] : !stream.timepoint
    scf.yield %dt0 : !stream.timepoint
  // CHECK: } else {
  } else {
    // CHECK: %[[IMMEDIATE:.+]] = stream.timepoint.immediate => !stream.timepoint
    %immediate = stream.timepoint.immediate => !stream.timepoint
    // CHECK: scf.yield %[[IMMEDIATE]] : !stream.timepoint
    scf.yield %immediate : !stream.timepoint
  // CHECK: }
  }

  %result, %result_tp = stream.resource.transients await(%final_tp) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // CHECK-NOT: stream.resource.alloca
  // CHECK-NOT: stream.resource.dealloca
  // CHECK-NOT: stream.resource.transients
  // CHECK: util.return %[[ARG0]], %[[ARG0_SIZE]]
  util.return %result, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests SCF control flow with transients in both branches.

// CHECK-LABEL: @scf_if_transient_in_both
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>, %[[STORAGE_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[COND:[a-z0-9]+]]: i1,
// CHECK-SAME:  %[[INPUT_TIMEPOINT:[a-z0-9]+]]: !stream.timepoint)
util.func public @scf_if_transient_in_both(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index,
  %cond: i1,
  %input_timepoint: !stream.timepoint
) -> (!stream.resource<*>, index) {
  // CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: %[[ELSE_BUFFER_SIZE:.+]] = arith.constant 256 : index
  %c256 = arith.constant 256 : index
  // CHECK-DAG: %[[THEN_BUFFER_SIZE:.+]] = arith.constant 512 : index
  %c512 = arith.constant 512 : index

  // Conservative max for mutually exclusive branches.
  // CHECK: %[[MAX0:.+]] = arith.maxui %[[ELSE_BUFFER_SIZE]], %[[ELSE_BUFFER_SIZE]] : index
  // CHECK: %[[MAX:.+]] = arith.maxui %[[MAX0]], %[[THEN_BUFFER_SIZE]] : index
  //
  // Single slot packing (both branches use same offset).
  // CHECK: %[[PACK:.+]]:2 = stream.resource.pack slices({
  // CHECK-NEXT:   [0, 0] = %[[MAX]]
  // CHECK-NEXT: }) : index attributes {stream.experimental.transients}
  //
  // CHECK: %[[BASE_OFFSET:.+]] = arith.constant 0 : index
  //
  // Storage subview.
  // CHECK: %[[STORAGE_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE]][%[[BASE_OFFSET]]] : !stream.resource<transient>{%[[STORAGE_SIZE]]} -> !stream.resource<transient>{%[[PACK]]#0}

  // Different transient allocations in each branch, both awaiting input timepoint.
  // CHECK: %[[FINAL_TP:.+]] = scf.if %[[COND]] -> (!stream.timepoint) {
  %final_tp = scf.if %cond -> !stream.timepoint {
    // Then branch subview uses original alloca size (not conservative max).
    // CHECK: %[[THEN_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_SUBVIEW]][%[[PACK]]#1] : !stream.resource<transient>{%[[PACK]]#0} -> !stream.resource<transient>{%[[THEN_BUFFER_SIZE]]}
    %t0, %tp0 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c512} => !stream.timepoint
    // CRITICAL: Then branch executes await input timepoint (alloca removed, timeline preserved).
    // CHECK: %[[THEN_EXEC:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[THEN_SUBVIEW]] as %[[THEN_CAP:.+]]: !stream.resource<transient>{%[[THEN_BUFFER_SIZE]]}) {
    // CHECK:   stream.cmd.fill %[[ZERO_I32]], %[[THEN_CAP]][%[[ZERO_INDEX]] for %[[THEN_BUFFER_SIZE]]] : i32 -> !stream.resource<transient>{%[[THEN_BUFFER_SIZE]]}
    // CHECK: } => !stream.timepoint
    %e0 = stream.cmd.execute await(%tp0) => with(%t0 as %c: !stream.resource<transient>{%c512}) {
      stream.cmd.fill %c0_i32, %c[%c0 for %c512] : i32 -> !stream.resource<transient>{%c512}
    } => !stream.timepoint
    %dt0 = stream.resource.dealloca await(%e0) => %t0 : !stream.resource<transient>{%c512} => !stream.timepoint
    // CHECK: scf.yield %[[THEN_EXEC]] : !stream.timepoint
    scf.yield %dt0 : !stream.timepoint
  // CHECK: } else {
  } else {
    // Else branch subview uses original alloca size (same offset due to slot sharing, but different size).
    // CHECK: %[[ELSE_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_SUBVIEW]][%[[PACK]]#1] : !stream.resource<transient>{%[[PACK]]#0} -> !stream.resource<transient>{%[[ELSE_BUFFER_SIZE]]}
    %t1, %tp1 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c256} => !stream.timepoint
    // CRITICAL: Else branch executes await input timepoint (alloca removed, timeline preserved).
    // CHECK: %[[ELSE_EXEC:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[ELSE_SUBVIEW]] as %[[ELSE_CAP:.+]]: !stream.resource<transient>{%[[ELSE_BUFFER_SIZE]]}) {
    // CHECK:   stream.cmd.fill %[[ZERO_I32]], %[[ELSE_CAP]][%[[ZERO_INDEX]] for %[[ELSE_BUFFER_SIZE]]] : i32 -> !stream.resource<transient>{%[[ELSE_BUFFER_SIZE]]}
    // CHECK: } => !stream.timepoint
    %e1 = stream.cmd.execute await(%tp1) => with(%t1 as %c: !stream.resource<transient>{%c256}) {
      stream.cmd.fill %c0_i32, %c[%c0 for %c256] : i32 -> !stream.resource<transient>{%c256}
    } => !stream.timepoint
    %dt1 = stream.resource.dealloca await(%e1) => %t1 : !stream.resource<transient>{%c256} => !stream.timepoint
    // CHECK: scf.yield %[[ELSE_EXEC]] : !stream.timepoint
    scf.yield %dt1 : !stream.timepoint
  // CHECK: }
  }

  %result, %result_tp = stream.resource.transients await(%final_tp) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // CHECK-NOT: stream.resource.alloca
  // CHECK-NOT: stream.resource.dealloca
  // CHECK-NOT: stream.resource.transients
  // CHECK: util.return %[[ARG0]], %[[ARG0_SIZE]]
  util.return %result, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests scf.for with loop-carried transient timepoint.

// CHECK-LABEL: @scf_for_with_transient
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>, %[[STORAGE_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[LB:[a-z0-9]+]]: index, %[[UB:[a-z0-9]+]]: index, %[[STEP:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[INPUT_TIMEPOINT:[a-z0-9]+]]: !stream.timepoint)
util.func public @scf_for_with_transient(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index,
  %lb: index, %ub: index, %step: index,
  %input_timepoint: !stream.timepoint
) -> (!stream.resource<*>, index) {
  // CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index

  // Loop with transient allocation inside, first iter awaits input timepoint.
  // CHECK: scf.for %{{.+}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[ITER_TP:.+]] = %[[INPUT_TIMEPOINT]]) -> (!stream.timepoint) {
  %final_tp = scf.for %iv = %lb to %ub step %step iter_args(%iter_tp = %input_timepoint) -> !stream.timepoint {
    // CHECK: %[[LOOP_BUFFER_SIZE:.+]] = arith.constant 128 : index
    // Pack with single slice for loop-reused transient (now inside loop).
    // CHECK: %[[PACK:.+]]:2 = stream.resource.pack slices({
    // CHECK-NEXT:   [0, 0] = %[[LOOP_BUFFER_SIZE]]
    // CHECK-NEXT: }) : index attributes {stream.experimental.transients}
    //
    // CHECK: %[[BASE_OFFSET:.+]] = arith.constant 0 : index
    //
    // Storage subview.
    // CHECK: %[[STORAGE_VIEW:.+]] = stream.resource.subview %[[STORAGE]][%[[BASE_OFFSET]]] : !stream.resource<transient>{%[[STORAGE_SIZE]]} -> !stream.resource<transient>{%[[PACK]]#0}
    // Transient subview created inside loop (once per iteration).
    // CHECK: %[[TRANSIENT_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_VIEW]][%[[PACK]]#1] : !stream.resource<transient>{%[[PACK]]#0} -> !stream.resource<transient>{%[[LOOP_BUFFER_SIZE]]}
    // CRITICAL: Join uses iter_tp which threads timeline through loop iterations.
    // CHECK: %[[JOINED:.+]] = stream.timepoint.join max(%[[ITER_TP]], %[[ITER_TP]]) => !stream.timepoint
    %t, %alloc_tp = stream.resource.alloca uninitialized await(%iter_tp) => !stream.resource<transient>{%c128} => !stream.timepoint
    %joined = stream.timepoint.join max(%iter_tp, %alloc_tp) => !stream.timepoint
    // CHECK: %[[EXEC_TP:.+]] = stream.cmd.execute await(%[[JOINED]]) => with(%[[TRANSIENT_SUBVIEW]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[LOOP_BUFFER_SIZE]]}) {
    // CHECK:   stream.cmd.fill %[[ZERO_I32]], %[[CAPTURE]][%[[ZERO_INDEX]] for %[[LOOP_BUFFER_SIZE]]] : i32 -> !stream.resource<transient>{%[[LOOP_BUFFER_SIZE]]}
    // CHECK: } => !stream.timepoint
    %exec_tp = stream.cmd.execute await(%joined) => with(%t as %c: !stream.resource<transient>{%c128}) {
      stream.cmd.fill %c0_i32, %c[%c0 for %c128] : i32 -> !stream.resource<transient>{%c128}
    } => !stream.timepoint
    %dealloc_tp = stream.resource.dealloca await(%exec_tp) => %t : !stream.resource<transient>{%c128} => !stream.timepoint
    // CHECK: scf.yield %[[EXEC_TP]] : !stream.timepoint
    scf.yield %dealloc_tp : !stream.timepoint
  // CHECK: }
  }

  %result, %result_tp = stream.resource.transients await(%final_tp) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // CHECK-NOT: stream.resource.alloca
  // CHECK-NOT: stream.resource.dealloca
  // CHECK-NOT: stream.resource.transients
  // CHECK: util.return %[[ARG0]], %[[ARG0_SIZE]]
  util.return %result, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests that size computation in scf.if is hoisted above the scf.if when alloca is later.

// CHECK-LABEL: @size_in_scf_if_alloca_later
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>, %[[STORAGE_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[COND:[a-z0-9]+]]: i1,
// CHECK-SAME:  %[[INPUT_TIMEPOINT:[a-z0-9]+]]: !stream.timepoint)
util.func public @size_in_scf_if_alloca_later(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index,
  %cond: i1,
  %input_timepoint: !stream.timepoint
) -> (!stream.resource<*>, index) {
  // CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: %[[THEN_BASE_SIZE:.+]] = arith.constant 512 : index
  %c512 = arith.constant 512 : index
  // CHECK-DAG: %[[ELSE_BASE_SIZE:.+]] = arith.constant 1024 : index
  %c1024 = arith.constant 1024 : index

  // Size computed in scf.if (original - used at execution).
  // CHECK: %[[TRANSIENT_SIZE:.+]] = scf.if %[[COND]] -> (index)
  %transient_size = scf.if %cond -> index {
    // CHECK: scf.yield %[[THEN_BASE_SIZE]] : index
    scf.yield %c512 : index
  } else {
    // CHECK: scf.yield %[[ELSE_BASE_SIZE]] : index
    scf.yield %c1024 : index
  }

  // Some other ops between scf.if and alloca.
  // CHECK: %[[MULTIPLIER:.+]] = arith.constant 2 : index
  %c2 = arith.constant 2 : index
  // CHECK: %[[ADJUSTED_SIZE:.+]] = arith.muli %[[TRANSIENT_SIZE]], %[[MULTIPLIER]]
  %adjusted_size = arith.muli %transient_size, %c2 : index

  // Cloned size computation for pack (hoisted after original).
  // CHECK: %[[TRANSIENT_SIZE_CLONE:.+]] = scf.if %[[COND]] -> (index)
  // CHECK: %[[MULTIPLIER_CLONE:.+]] = arith.constant 2 : index
  // CHECK: %[[ADJUSTED_SIZE_CLONE:.+]] = arith.muli %[[TRANSIENT_SIZE_CLONE]], %[[MULTIPLIER_CLONE]]
  //
  // Pack uses cloned computation.
  // CHECK: %[[PACK:.+]]:2 = stream.resource.pack slices({
  // CHECK-NEXT: [0, 0] = %[[ADJUSTED_SIZE_CLONE]]
  // CHECK-NEXT: }) : index attributes {stream.experimental.transients}
  //
  // CHECK: %[[BASE_OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[STORAGE_VIEW:.+]] = stream.resource.subview %[[STORAGE]][%[[BASE_OFFSET]]] : !stream.resource<transient>{%[[STORAGE_SIZE]]} -> !stream.resource<transient>{%[[PACK]]#0}

  // Allocate transient with size from scf.if, awaiting input timepoint.
  // CHECK: %[[TRANSIENT_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_VIEW]][%[[PACK]]#1]
  %transient, %alloca_timepoint = stream.resource.alloca uninitialized await(%input_timepoint) =>
      !stream.resource<transient>{%adjusted_size} => !stream.timepoint
  // Subview from alloca replacement flows into execute.
  // CHECK: %[[EXEC_TP:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[TRANSIENT_SUBVIEW]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[ADJUSTED_SIZE]]}) {
  // CHECK:   stream.cmd.fill %[[ZERO_I32]], %[[CAPTURE]][%[[ZERO_INDEX]] for %[[ADJUSTED_SIZE]]] : i32 -> !stream.resource<transient>{%[[ADJUSTED_SIZE]]}
  // CHECK: } => !stream.timepoint
  %exec_timepoint = stream.cmd.execute
      await(%alloca_timepoint) => with(%transient as %t: !stream.resource<transient>{%adjusted_size}) {
    stream.cmd.fill %c0_i32, %t[%c0 for %adjusted_size] : i32 -> !stream.resource<transient>{%adjusted_size}
  } => !stream.timepoint

  %dealloca_timepoint = stream.resource.dealloca await(%exec_timepoint) => %transient : !stream.resource<transient>{%adjusted_size} => !stream.timepoint

  %result_annotated, %result_tp = stream.resource.transients await(%dealloca_timepoint) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // CHECK-NOT: stream.resource.alloca
  // CHECK-NOT: stream.resource.dealloca
  // CHECK-NOT: stream.resource.transients
  // CHECK: util.return %[[ARG0]], %[[ARG0_SIZE]]
  util.return %result_annotated, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests that alloca in scf.for with size computed outside is handled.

// CHECK-LABEL: @alloca_in_scf_for_size_outside
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>, %[[STORAGE_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[LB:[a-z0-9]+]]: index, %[[UB:[a-z0-9]+]]: index, %[[STEP:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[INPUT_TIMEPOINT:[a-z0-9]+]]: !stream.timepoint)
util.func public @alloca_in_scf_for_size_outside(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index,
  %lb: index, %ub: index, %step: index,
  %input_timepoint: !stream.timepoint
) -> (!stream.resource<*>, index) {
  // CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: %[[BYTES_PER_ELEMENT:.+]] = arith.constant 4 : index
  %c4 = arith.constant 4 : index

  // Size computation already outside loop - reused directly.
  %loop_size = arith.muli %arg0_size, %c4 : index

  // Alloca inside loop with size from outside, first iter awaits input timepoint.
  // scf.for with pack and subview created inside loop.
  // CHECK: scf.for %{{.+}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[ITER_TP:.+]] = %[[INPUT_TIMEPOINT]]) -> (!stream.timepoint) {
  %final_tp = scf.for %iv = %lb to %ub step %step iter_args(%iter_tp = %input_timepoint) -> !stream.timepoint {
    // Size computation moved inside loop.
    // CHECK: %[[LOOP_SIZE_INNER:.+]] = arith.muli %[[ARG0_SIZE]], %[[BYTES_PER_ELEMENT]] : index
    // Pack now inside loop.
    // CHECK: %[[PACK:.+]]:2 = stream.resource.pack slices({
    // CHECK-NEXT:   [0, 0] = %[[LOOP_SIZE_INNER]]
    // CHECK-NEXT: }) : index attributes {stream.experimental.transients}
    // CHECK: %[[BASE_OFFSET:.+]] = arith.constant 0 : index
    // CHECK: %[[STORAGE_VIEW:.+]] = stream.resource.subview %[[STORAGE]][%[[BASE_OFFSET]]] : !stream.resource<transient>{%[[STORAGE_SIZE]]} -> !stream.resource<transient>{%[[PACK]]#0}
    // CHECK: %[[TRANSIENT_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_VIEW]][%[[PACK]]#1]
    %t, %alloc_tp = stream.resource.alloca uninitialized await(%iter_tp) => !stream.resource<transient>{%loop_size} => !stream.timepoint
    // CHECK: %[[JOINED:.+]] = stream.timepoint.join max(%[[ITER_TP]], %[[ITER_TP]]) => !stream.timepoint
    %joined = stream.timepoint.join max(%iter_tp, %alloc_tp) => !stream.timepoint
    // CHECK: %[[EXEC_TP:.+]] = stream.cmd.execute await(%[[JOINED]]) => with(%[[TRANSIENT_SUBVIEW]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[LOOP_SIZE_INNER]]}) {
    // CHECK:   stream.cmd.fill %[[ZERO_I32]], %[[CAPTURE]][%[[ZERO_INDEX]] for %[[LOOP_SIZE_INNER]]] : i32 -> !stream.resource<transient>{%[[LOOP_SIZE_INNER]]}
    // CHECK: } => !stream.timepoint
    %exec_tp = stream.cmd.execute await(%joined) => with(%t as %c: !stream.resource<transient>{%loop_size}) {
      stream.cmd.fill %c0_i32, %c[%c0 for %loop_size] : i32 -> !stream.resource<transient>{%loop_size}
    } => !stream.timepoint
    %dealloc_tp = stream.resource.dealloca await(%exec_tp) => %t : !stream.resource<transient>{%loop_size} => !stream.timepoint
    // CHECK: scf.yield %[[EXEC_TP]] : !stream.timepoint
    scf.yield %dealloc_tp : !stream.timepoint
  }

  %result, %result_tp = stream.resource.transients await(%final_tp) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // CHECK-NOT: stream.resource.alloca
  // CHECK-NOT: stream.resource.dealloca
  // CHECK-NOT: stream.resource.transients
  // CHECK: util.return %[[ARG0]], %[[ARG0_SIZE]]
  util.return %result, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests hoisting with multiple levels of SCF nesting.

// CHECK-LABEL: @nested_scf_regions
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>, %[[STORAGE_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[COND1:[a-z0-9]+]]: i1, %[[COND2:[a-z0-9]+]]: i1,
// CHECK-SAME:  %[[INPUT_TIMEPOINT:[a-z0-9]+]]: !stream.timepoint)
util.func public @nested_scf_regions(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index,
  %cond1: i1, %cond2: i1,
  %input_timepoint: !stream.timepoint
) -> (!stream.resource<*>, index) {
  // CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  %c256 = arith.constant 256 : index
  // CHECK-DAG: %[[INNER_THEN_SIZE:.+]] = arith.constant 512 : index
  %c512 = arith.constant 512 : index
  // CHECK-DAG: %[[INNER_ELSE_SIZE:.+]] = arith.constant 1024 : index
  %c1024 = arith.constant 1024 : index

  // Outer scf.if containing allocation logic.
  // CHECK: scf.if %[[COND1]] -> (!stream.timepoint)
  %final_tp = scf.if %cond1 -> !stream.timepoint {
    // Compute size in outer scf.if.
    // Inner scf.if for size (original - used at execution).
    // CHECK: %[[OUTER_SIZE:.+]] = scf.if %[[COND2]] -> (index)
    %outer_size = scf.if %cond2 -> index {
      // CHECK: scf.yield %[[INNER_THEN_SIZE]] : index
      scf.yield %c512 : index
    } else {
      // CHECK: scf.yield %[[INNER_ELSE_SIZE]] : index
      scf.yield %c1024 : index
    }

    // Cloned inner scf.if for pack (hoisted within outer branch).
    // CHECK: %[[OUTER_SIZE_CLONE:.+]] = scf.if %[[COND2]] -> (index)
    //
    // Pack inside outer then branch (but before execution).
    // CHECK: %[[PACK:.+]]:2 = stream.resource.pack slices({
    // CHECK-NEXT:   [0, 0] = %[[OUTER_SIZE_CLONE]]
    // CHECK-NEXT: }) : index attributes {stream.experimental.transients}

    // CHECK: %[[BASE_OFFSET:.+]] = arith.constant 0 : index
    // CHECK: %[[STORAGE_VIEW:.+]] = stream.resource.subview %[[STORAGE]][%[[BASE_OFFSET]]] : !stream.resource<transient>{%[[STORAGE_SIZE]]} -> !stream.resource<transient>{%[[PACK]]#0}

    // Allocate in nested scf.if with size from above, awaiting input timepoint.
    // CHECK: %[[TRANSIENT_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_VIEW]][%[[PACK]]#1]
    %t, %alloc_tp = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%outer_size} => !stream.timepoint
    // CRITICAL: Execute awaits input timepoint (alloca removed, timeline preserved).
    // Subview from alloca replacement flows into execute.
    // CHECK: %[[EXEC_TP:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[TRANSIENT_SUBVIEW]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[OUTER_SIZE]]}) {
    // CHECK:   stream.cmd.fill %[[ZERO_I32]], %[[CAPTURE]][%[[ZERO_INDEX]] for %[[OUTER_SIZE]]] : i32 -> !stream.resource<transient>{%[[OUTER_SIZE]]}
    // CHECK: } => !stream.timepoint
    %exec_tp = stream.cmd.execute await(%alloc_tp) => with(%t as %c: !stream.resource<transient>{%outer_size}) {
      stream.cmd.fill %c0_i32, %c[%c0 for %outer_size] : i32 -> !stream.resource<transient>{%outer_size}
    } => !stream.timepoint
    %dealloc_tp = stream.resource.dealloca await(%exec_tp) => %t : !stream.resource<transient>{%outer_size} => !stream.timepoint
    scf.yield %dealloc_tp : !stream.timepoint
  } else {
    %imm = stream.timepoint.immediate => !stream.timepoint
    scf.yield %imm : !stream.timepoint
  }

  %result, %result_tp = stream.resource.transients await(%final_tp) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // CHECK-NOT: stream.resource.alloca
  // CHECK-NOT: stream.resource.dealloca
  // CHECK-NOT: stream.resource.transients
  // CHECK: util.return %[[ARG0]], %[[ARG0_SIZE]]
  util.return %result, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests nested scf.if with transient allocations at multiple levels.
// This stresses the region ancestry checks and conservative max computation.

// CHECK-LABEL: @nested_scf_if
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>, %[[STORAGE_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[OUTER_COND:[a-z0-9]+]]: i1, %[[INNER_COND:[a-z0-9]+]]: i1,
// CHECK-SAME:  %[[INPUT_TIMEPOINT:[a-z0-9]+]]: !stream.timepoint)
util.func public @nested_scf_if(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index,
  %outer_cond: i1, %inner_cond: i1,
  %input_timepoint: !stream.timepoint
) -> (!stream.resource<*>, index) {
  // CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %c1024 = arith.constant 1024 : index

  // Conservative max for nested inner if (512 vs 1024) - constants hoisted here.
  // CHECK: %[[INNER_ELSE_SIZE:.+]] = arith.constant 512 : index
  // CHECK: %[[INNER_THEN_SIZE:.+]] = arith.constant 1024 : index
  // CHECK: %[[MAX0:.+]] = arith.maxui %[[INNER_ELSE_SIZE]], %[[INNER_ELSE_SIZE]] : index
  // CHECK: %[[MAX1:.+]] = arith.maxui %[[MAX0]], %[[INNER_THEN_SIZE]] : index
  // CHECK: %[[OUTER_ELSE_SIZE:.+]] = arith.constant 128 : index
  //
  // Pack with 2 slots: slot0 for nested if branches, slot1 for outer else.
  // CHECK: %[[PACK:.+]]:3 = stream.resource.pack slices({
  // CHECK-NEXT:   [0, 0] = %[[MAX1]],
  // CHECK-NEXT:   [0, 0] = %[[OUTER_ELSE_SIZE]]
  // CHECK-NEXT: }) : index attributes {stream.experimental.transients}
  // CHECK: %[[BASE_OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[STORAGE_VIEW:.+]] = stream.resource.subview %[[STORAGE]][%[[BASE_OFFSET]]] : !stream.resource<transient>{%[[STORAGE_SIZE]]} -> !stream.resource<transient>{%[[PACK]]#0}

  // CHECK: %{{.+}} = scf.if %[[OUTER_COND]]
  %final_tp = scf.if %outer_cond -> !stream.timepoint {
    // CHECK: %{{.+}} = scf.if %[[INNER_COND]]
    %nested_tp = scf.if %inner_cond -> !stream.timepoint {
      // CHECK: %{{.+}} = stream.resource.subview %[[STORAGE_VIEW]][%[[PACK]]#1]
      // CHECK: stream.cmd.execute await(%[[INPUT_TIMEPOINT]])
      %t0, %tp0 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c1024} => !stream.timepoint
      %e0 = stream.cmd.execute await(%tp0) => with(%t0 as %c: !stream.resource<transient>{%c1024}) {
        stream.cmd.fill %c0_i32, %c[%c0 for %c1024] : i32 -> !stream.resource<transient>{%c1024}
      } => !stream.timepoint
      %dt0 = stream.resource.dealloca await(%e0) => %t0 : !stream.resource<transient>{%c1024} => !stream.timepoint
      scf.yield %dt0 : !stream.timepoint
    } else {
      %t1, %tp1 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c512} => !stream.timepoint
      %e1 = stream.cmd.execute await(%tp1) => with(%t1 as %c: !stream.resource<transient>{%c512}) {
        stream.cmd.fill %c0_i32, %c[%c0 for %c512] : i32 -> !stream.resource<transient>{%c512}
      } => !stream.timepoint
      %dt1 = stream.resource.dealloca await(%e1) => %t1 : !stream.resource<transient>{%c512} => !stream.timepoint
      scf.yield %dt1 : !stream.timepoint
    }
    scf.yield %nested_tp : !stream.timepoint
  } else {
    // CHECK: stream.cmd.execute
    // Outer else branch with different size.
    %t2, %tp2 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c128} => !stream.timepoint
    %e2 = stream.cmd.execute await(%tp2) => with(%t2 as %c: !stream.resource<transient>{%c128}) {
      stream.cmd.fill %c0_i32, %c[%c0 for %c128] : i32 -> !stream.resource<transient>{%c128}
    } => !stream.timepoint
    %dt2 = stream.resource.dealloca await(%e2) => %t2 : !stream.resource<transient>{%c128} => !stream.timepoint
    scf.yield %dt2 : !stream.timepoint
  }

  %result, %result_tp = stream.resource.transients await(%final_tp) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // CHECK-NOT: stream.resource.alloca
  // CHECK-NOT: stream.resource.dealloca
  // CHECK-NOT: stream.resource.transients
  // CHECK: util.return %[[ARG0]], %[[ARG0_SIZE]]
  util.return %result, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests scf.for with nested scf.if containing transient allocations.
// This tests mixed control flow nesting.

// CHECK-LABEL: @scf_for_with_nested_if
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>, %[[STORAGE_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[LB:[a-z0-9]+]]: index, %[[UB:[a-z0-9]+]]: index, %[[STEP:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[COND:[a-z0-9]+]]: i1,
// CHECK-SAME:  %[[INPUT_TIMEPOINT:[a-z0-9]+]]: !stream.timepoint)
util.func public @scf_for_with_nested_if(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index,
  %lb: index, %ub: index, %step: index,
  %cond: i1,
  %input_timepoint: !stream.timepoint
) -> (!stream.resource<*>, index) {
  // CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: %[[ELSE_BUFFER_SIZE:.+]] = arith.constant 256 : index
  %c256 = arith.constant 256 : index
  // CHECK-DAG: %[[THEN_BUFFER_SIZE:.+]] = arith.constant 512 : index
  %c512 = arith.constant 512 : index

  // Conservative max for scf.if branches - partially outside loop.
  // CHECK: %[[MAX0:.+]] = arith.maxui %[[ELSE_BUFFER_SIZE]], %[[ELSE_BUFFER_SIZE]] : index
  // Single slot packing will be inside loop.
  //
  // scf.for with nested scf.if, both awaiting input timepoint.
  // CHECK: scf.for %{{.+}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[ITER_TP:.+]] = %[[INPUT_TIMEPOINT]]) -> (!stream.timepoint) {
  %final_tp = scf.for %iv = %lb to %ub step %step iter_args(%iter_tp = %input_timepoint) -> !stream.timepoint {
    // Rest of conservative max inside loop.
    // CHECK: %[[MAX:.+]] = arith.maxui %[[MAX0]], %[[THEN_BUFFER_SIZE]] : index
    // CHECK: %[[PACK:.+]]:2 = stream.resource.pack slices({
    // CHECK-NEXT:   [0, 0] = %[[MAX]]
    // CHECK-NEXT: }) : index attributes {stream.experimental.transients}
    // CHECK: %[[BASE_OFFSET:.+]] = arith.constant 0 : index
    // CHECK: %[[STORAGE_VIEW:.+]] = stream.resource.subview %[[STORAGE]][%[[BASE_OFFSET]]] : !stream.resource<transient>{%[[STORAGE_SIZE]]} -> !stream.resource<transient>{%[[PACK]]#0}
    // CHECK: %{{.+}} = scf.if %[[COND]]
    %branch_tp = scf.if %cond -> !stream.timepoint {
      // CHECK: %{{.+}} = stream.resource.subview %[[STORAGE_VIEW]][%[[PACK]]#1]
      %t0, %tp0 = stream.resource.alloca uninitialized await(%iter_tp) => !stream.resource<transient>{%c512} => !stream.timepoint
      // CHECK: stream.timepoint.join
      %joined0 = stream.timepoint.join max(%iter_tp, %tp0) => !stream.timepoint
      // CHECK: stream.cmd.execute
      %e0 = stream.cmd.execute await(%joined0) => with(%t0 as %c: !stream.resource<transient>{%c512}) {
        stream.cmd.fill %c0_i32, %c[%c0 for %c512] : i32 -> !stream.resource<transient>{%c512}
      } => !stream.timepoint
      %dt0 = stream.resource.dealloca await(%e0) => %t0 : !stream.resource<transient>{%c512} => !stream.timepoint
      scf.yield %dt0 : !stream.timepoint
    } else {
      %t1, %tp1 = stream.resource.alloca uninitialized await(%iter_tp) => !stream.resource<transient>{%c256} => !stream.timepoint
      %joined1 = stream.timepoint.join max(%iter_tp, %tp1) => !stream.timepoint
      %e1 = stream.cmd.execute await(%joined1) => with(%t1 as %c: !stream.resource<transient>{%c256}) {
        stream.cmd.fill %c0_i32, %c[%c0 for %c256] : i32 -> !stream.resource<transient>{%c256}
      } => !stream.timepoint
      %dt1 = stream.resource.dealloca await(%e1) => %t1 : !stream.resource<transient>{%c256} => !stream.timepoint
      scf.yield %dt1 : !stream.timepoint
    }
    scf.yield %branch_tp : !stream.timepoint
  }

  %result, %result_tp = stream.resource.transients await(%final_tp) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // CHECK-NOT: stream.resource.alloca
  // CHECK-NOT: stream.resource.dealloca
  // CHECK-NOT: stream.resource.transients
  // CHECK: util.return %[[ARG0]], %[[ARG0_SIZE]]
  util.return %result, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests multiple allocations in each branch with different sizes.
// This stresses the conservative max computation with many values.

// CHECK-LABEL: @scf_if_multiple_allocations_per_branch
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>, %[[STORAGE_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[COND:[a-z0-9]+]]: i1,
// CHECK-SAME:  %[[INPUT_TIMEPOINT:[a-z0-9]+]]: !stream.timepoint)
util.func public @scf_if_multiple_allocations_per_branch(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index,
  %cond: i1,
  %input_timepoint: !stream.timepoint
) -> (!stream.resource<*>, index) {
  // CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: %[[ELSE_FIRST_SIZE:.+]] = arith.constant 128 : index
  %c128 = arith.constant 128 : index
  // CHECK-DAG: %[[ELSE_SECOND_SIZE:.+]] = arith.constant 256 : index
  %c256 = arith.constant 256 : index
  // CHECK-DAG: %[[THEN_FIRST_SIZE:.+]] = arith.constant 512 : index
  %c512 = arith.constant 512 : index

  // Conservative max computations for slot 0 (128 vs 512).
  // CHECK: %[[SLOT0_MAX0:.+]] = arith.maxui %[[ELSE_FIRST_SIZE]], %[[ELSE_FIRST_SIZE]] : index
  // CHECK: %[[THEN_SECOND_SIZE:.+]] = arith.constant 1024 : index
  %c1024 = arith.constant 1024 : index
  // Conservative max computations for slot 1 (256 vs 1024).
  // CHECK: %[[SLOT1_MAX0:.+]] = arith.maxui %[[ELSE_SECOND_SIZE]], %[[ELSE_SECOND_SIZE]] : index
  // CHECK: %[[SLOT0_MAX:.+]] = arith.maxui %[[SLOT0_MAX0]], %[[THEN_FIRST_SIZE]] : index
  // CHECK: %[[SLOT1_MAX:.+]] = arith.maxui %[[SLOT1_MAX0]], %[[THEN_SECOND_SIZE]]
  //
  // Pack with 2 slots for 2 allocations per branch.
  // CHECK: %[[PACK:.+]]:3 = stream.resource.pack slices({
  // CHECK-NEXT:   [0, 0] = %[[SLOT0_MAX]],
  // CHECK-NEXT:   [0, 0] = %[[SLOT1_MAX]]
  // CHECK-NEXT: }) : index attributes {stream.experimental.transients}

  // CHECK: %[[BASE_OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[STORAGE_VIEW:.+]] = stream.resource.subview %[[STORAGE]][%[[BASE_OFFSET]]] : !stream.resource<transient>{%[[STORAGE_SIZE]]} -> !stream.resource<transient>{%[[PACK]]#0}

  // scf.if with both branches using subviews, joins at end.
  // CHECK: scf.if %[[COND]]
  %final_tp = scf.if %cond -> !stream.timepoint {
    // Then branch: two allocations awaiting input timepoint.
    // CHECK: %[[THEN_SUB0:.+]] = stream.resource.subview %[[STORAGE_VIEW]][%[[PACK]]#1]
    %t0, %tp0 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c512} => !stream.timepoint
    // CHECK: %[[THEN_EXEC0:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[THEN_SUB0]] as %[[THEN_CAP0:.+]]: !stream.resource<transient>{%[[THEN_FIRST_SIZE]]}) {
    // CHECK:   stream.cmd.fill %[[ZERO_I32]], %[[THEN_CAP0]][%[[ZERO_INDEX]] for %[[THEN_FIRST_SIZE]]] : i32 -> !stream.resource<transient>{%[[THEN_FIRST_SIZE]]}
    // CHECK: } => !stream.timepoint
    %e0 = stream.cmd.execute await(%tp0) => with(%t0 as %c: !stream.resource<transient>{%c512}) {
      stream.cmd.fill %c0_i32, %c[%c0 for %c512] : i32 -> !stream.resource<transient>{%c512}
    } => !stream.timepoint
    %dt0 = stream.resource.dealloca await(%e0) => %t0 : !stream.resource<transient>{%c512} => !stream.timepoint

    // CHECK: %[[THEN_SUB1:.+]] = stream.resource.subview %[[STORAGE_VIEW]][%[[PACK]]#2]
    %t1, %tp1 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c1024} => !stream.timepoint
    // CHECK: %[[THEN_EXEC1:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[THEN_SUB1]] as %[[THEN_CAP1:.+]]: !stream.resource<transient>{%[[THEN_SECOND_SIZE]]}) {
    // CHECK:   stream.cmd.fill %[[ZERO_I32]], %[[THEN_CAP1]][%[[ZERO_INDEX]] for %[[THEN_SECOND_SIZE]]] : i32 -> !stream.resource<transient>{%[[THEN_SECOND_SIZE]]}
    // CHECK: } => !stream.timepoint
    %e1 = stream.cmd.execute await(%tp1) => with(%t1 as %c: !stream.resource<transient>{%c1024}) {
      stream.cmd.fill %c0_i32, %c[%c0 for %c1024] : i32 -> !stream.resource<transient>{%c1024}
    } => !stream.timepoint
    %dt1 = stream.resource.dealloca await(%e1) => %t1 : !stream.resource<transient>{%c1024} => !stream.timepoint

    // CHECK: %[[THEN_JOINED:.+]] = stream.timepoint.join max(%[[THEN_EXEC0]], %[[THEN_EXEC1]]) => !stream.timepoint
    %joined = stream.timepoint.join max(%dt0, %dt1) => !stream.timepoint
    // CHECK: scf.yield %[[THEN_JOINED]] : !stream.timepoint
    scf.yield %joined : !stream.timepoint
  } else {
    // Else branch: two different allocations awaiting input timepoint.
    // CHECK: %[[ELSE_SUB0:.+]] = stream.resource.subview %[[STORAGE_VIEW]][%[[PACK]]#1]
    %t2, %tp2 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c128} => !stream.timepoint
    // CHECK: %[[ELSE_EXEC0:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[ELSE_SUB0]] as %[[ELSE_CAP0:.+]]: !stream.resource<transient>{%[[ELSE_FIRST_SIZE]]}) {
    // CHECK:   stream.cmd.fill %[[ZERO_I32]], %[[ELSE_CAP0]][%[[ZERO_INDEX]] for %[[ELSE_FIRST_SIZE]]] : i32 -> !stream.resource<transient>{%[[ELSE_FIRST_SIZE]]}
    // CHECK: } => !stream.timepoint
    %e2 = stream.cmd.execute await(%tp2) => with(%t2 as %c: !stream.resource<transient>{%c128}) {
      stream.cmd.fill %c0_i32, %c[%c0 for %c128] : i32 -> !stream.resource<transient>{%c128}
    } => !stream.timepoint
    %dt2 = stream.resource.dealloca await(%e2) => %t2 : !stream.resource<transient>{%c128} => !stream.timepoint

    // CHECK: %[[ELSE_SUB1:.+]] = stream.resource.subview %[[STORAGE_VIEW]][%[[PACK]]#2]
    %t3, %tp3 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c256} => !stream.timepoint
    // CHECK: %[[ELSE_EXEC1:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[ELSE_SUB1]] as %[[ELSE_CAP1:.+]]: !stream.resource<transient>{%[[ELSE_SECOND_SIZE]]}) {
    // CHECK:   stream.cmd.fill %[[ZERO_I32]], %[[ELSE_CAP1]][%[[ZERO_INDEX]] for %[[ELSE_SECOND_SIZE]]] : i32 -> !stream.resource<transient>{%[[ELSE_SECOND_SIZE]]}
    // CHECK: } => !stream.timepoint
    %e3 = stream.cmd.execute await(%tp3) => with(%t3 as %c: !stream.resource<transient>{%c256}) {
      stream.cmd.fill %c0_i32, %c[%c0 for %c256] : i32 -> !stream.resource<transient>{%c256}
    } => !stream.timepoint
    %dt3 = stream.resource.dealloca await(%e3) => %t3 : !stream.resource<transient>{%c256} => !stream.timepoint

    // CHECK: %[[ELSE_JOINED:.+]] = stream.timepoint.join max(%[[ELSE_EXEC0]], %[[ELSE_EXEC1]]) => !stream.timepoint
    %joined = stream.timepoint.join max(%dt2, %dt3) => !stream.timepoint
    // CHECK: scf.yield %[[ELSE_JOINED]] : !stream.timepoint
    scf.yield %joined : !stream.timepoint
  }

  %result, %result_tp = stream.resource.transients await(%final_tp) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // CHECK-NOT: stream.resource.alloca
  // CHECK-NOT: stream.resource.dealloca
  // CHECK-NOT: stream.resource.transients
  // CHECK: util.return %[[ARG0]], %[[ARG0_SIZE]]
  util.return %result, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests computed sizes (not just constants) in mutually exclusive branches.
// This verifies that backward slicing handles arithmetic operations across
// region boundaries by hoisting them to a common dominator.

// CHECK-LABEL: @scf_if_computed_sizes
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>, %[[STORAGE_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[COND:[a-z0-9]+]]: i1,
// CHECK-SAME:  %[[BASE_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[INPUT_TIMEPOINT:[a-z0-9]+]]: !stream.timepoint)
util.func public @scf_if_computed_sizes(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index,
  %cond: i1,
  %base_size: index,
  %input_timepoint: !stream.timepoint
) -> (!stream.resource<*>, index) {
  // CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: %[[ELSE_MULTIPLIER:.+]] = arith.constant 2 : index
  %c2 = arith.constant 2 : index
  // CHECK-DAG: %[[THEN_MULTIPLIER:.+]] = arith.constant 4 : index
  %c4 = arith.constant 4 : index

  // CRITICAL: Cross-region hoisting - arith.muli ops hoisted BEFORE scf.if.
  // CHECK: %[[SIZE_ELSE:.+]] = arith.muli %[[BASE_SIZE]], %[[ELSE_MULTIPLIER]] : index
  // CHECK: %[[SIZE_THEN:.+]] = arith.muli %[[BASE_SIZE]], %[[THEN_MULTIPLIER]] : index
  //
  // Conservative max using hoisted computed sizes.
  // CHECK: %[[MAX0:.+]] = arith.maxui %[[SIZE_ELSE]], %[[SIZE_ELSE]] : index
  // CHECK: %[[MAX:.+]] = arith.maxui %[[MAX0]], %[[SIZE_THEN]] : index
  //
  // Single slot packing with conservative max size.
  // CHECK: %[[PACK:.+]]:2 = stream.resource.pack slices({
  // CHECK-NEXT: [0, 0] = %[[MAX]]
  // CHECK-NEXT: }) : index attributes {stream.experimental.transients}
  // CHECK: %[[BASE_OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[STORAGE_VIEW:.+]] = stream.resource.subview %[[STORAGE]][%[[BASE_OFFSET]]] : !stream.resource<transient>{%[[STORAGE_SIZE]]} -> !stream.resource<transient>{%[[PACK]]#0}

  // scf.if with recomputed sizes inside branches, both await input timepoint.
  // CHECK: scf.if %[[COND]]
  %final_tp = scf.if %cond -> !stream.timepoint {
    // Then branch: base_size * 4, awaiting input timepoint.
    // Recomputed size for execution (original used for pack).
    // CHECK: %[[SIZE_THEN_RECOMP:.+]] = arith.muli %[[BASE_SIZE]], %[[THEN_MULTIPLIER]] : index
    %size0 = arith.muli %base_size, %c4 : index
    // CHECK: %[[THEN_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_VIEW]][%[[PACK]]#1]
    %t0, %tp0 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%size0} => !stream.timepoint
    // CHECK: %[[THEN_EXEC:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[THEN_SUBVIEW]] as %[[THEN_CAP:.+]]: !stream.resource<transient>{%[[SIZE_THEN_RECOMP]]}) {
    // CHECK:   stream.cmd.fill %[[ZERO_I32]], %[[THEN_CAP]][%[[ZERO_INDEX]] for %[[SIZE_THEN_RECOMP]]] : i32 -> !stream.resource<transient>{%[[SIZE_THEN_RECOMP]]}
    // CHECK: } => !stream.timepoint
    %e0 = stream.cmd.execute await(%tp0) => with(%t0 as %c: !stream.resource<transient>{%size0}) {
      stream.cmd.fill %c0_i32, %c[%c0 for %size0] : i32 -> !stream.resource<transient>{%size0}
    } => !stream.timepoint
    %dt0 = stream.resource.dealloca await(%e0) => %t0 : !stream.resource<transient>{%size0} => !stream.timepoint
    // CHECK: scf.yield %[[THEN_EXEC]] : !stream.timepoint
    scf.yield %dt0 : !stream.timepoint
  // CHECK: } else {
  } else {
    // Else branch: base_size * 2, awaiting input timepoint.
    // Recomputed size for execution.
    // CHECK: %[[SIZE_ELSE_RECOMP:.+]] = arith.muli %[[BASE_SIZE]], %[[ELSE_MULTIPLIER]] : index
    %size1 = arith.muli %base_size, %c2 : index
    // CHECK: %[[ELSE_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_VIEW]][%[[PACK]]#1]
    %t1, %tp1 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%size1} => !stream.timepoint
    // CHECK: %[[ELSE_EXEC:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[ELSE_SUBVIEW]] as %[[ELSE_CAP:.+]]: !stream.resource<transient>{%[[SIZE_ELSE_RECOMP]]}) {
    // CHECK:   stream.cmd.fill %[[ZERO_I32]], %[[ELSE_CAP]][%[[ZERO_INDEX]] for %[[SIZE_ELSE_RECOMP]]] : i32 -> !stream.resource<transient>{%[[SIZE_ELSE_RECOMP]]}
    // CHECK: } => !stream.timepoint
    %e1 = stream.cmd.execute await(%tp1) => with(%t1 as %c: !stream.resource<transient>{%size1}) {
      stream.cmd.fill %c0_i32, %c[%c0 for %size1] : i32 -> !stream.resource<transient>{%size1}
    } => !stream.timepoint
    %dt1 = stream.resource.dealloca await(%e1) => %t1 : !stream.resource<transient>{%size1} => !stream.timepoint
    // CHECK: scf.yield %[[ELSE_EXEC]] : !stream.timepoint
    scf.yield %dt1 : !stream.timepoint
  }

  %result, %result_tp = stream.resource.transients await(%final_tp) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // CHECK-NOT: stream.resource.alloca
  // CHECK-NOT: stream.resource.dealloca
  // CHECK-NOT: stream.resource.transients
  // CHECK: util.return %[[ARG0]], %[[ARG0_SIZE]]
  util.return %result, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests deeply nested control flow: scf.if -> scf.if -> scf.if.
// This stresses the region ancestry tracking with multiple levels.

// CHECK-LABEL: @deeply_nested_scf_if
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>, %[[STORAGE_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[COND1:[a-z0-9]+]]: i1, %[[COND2:[a-z0-9]+]]: i1, %[[COND3:[a-z0-9]+]]: i1,
// CHECK-SAME:  %[[INPUT_TIMEPOINT:[a-z0-9]+]]: !stream.timepoint)
util.func public @deeply_nested_scf_if(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index,
  %cond1: i1, %cond2: i1, %cond3: i1,
  %input_timepoint: !stream.timepoint
) -> (!stream.resource<*>, index) {
  // CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  // CHECK: %[[INNER_ELSE_SIZE:.+]] = arith.constant 256 : index
  %c256 = arith.constant 256 : index
  // CHECK: %[[INNER_THEN_SIZE:.+]] = arith.constant 512 : index
  %c512 = arith.constant 512 : index

  // Conservative max for innermost nested if (level 3: 256 vs 512).
  // CHECK: %[[INNER_MAX0:.+]] = arith.maxui %[[INNER_ELSE_SIZE]], %[[INNER_ELSE_SIZE]] : index
  // CHECK: %[[INNER_MAX:.+]] = arith.maxui %[[INNER_MAX0]], %[[INNER_THEN_SIZE]] : index
  // CHECK: %[[OUTER_ELSE_SIZE:.+]] = arith.constant 64 : index
  // CHECK: %[[MIDDLE_ELSE_SIZE:.+]] = arith.constant 128 : index
  //
  // Pack with 3 slots for 3 distinct nesting levels.
  // CHECK: %[[PACK:.+]]:4 = stream.resource.pack slices({
  // CHECK-NEXT: [0, 0] = %[[INNER_MAX]],
  // CHECK-NEXT: [0, 0] = %[[OUTER_ELSE_SIZE]],
  // CHECK-NEXT: [0, 0] = %[[MIDDLE_ELSE_SIZE]]
  // CHECK-NEXT: }) : index attributes {stream.experimental.transients}
  //
  // Nested scf.if structure (3 levels).
  // CHECK: scf.if %[[COND1]]
  %final_tp = scf.if %cond1 -> !stream.timepoint {
    // CHECK: scf.if %[[COND2]]
    %l1_tp = scf.if %cond2 -> !stream.timepoint {
      // CHECK: scf.if %[[COND3]]
      %l2_tp = scf.if %cond3 -> !stream.timepoint {
        // Level 3: innermost, awaiting input timepoint.
        // Slot 0 subviews for innermost if branches (same offset).
        // CHECK: stream.resource.subview %{{.+}}[%[[PACK]]#1]
        %t0, %tp0 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c512} => !stream.timepoint
        // CHECK: stream.cmd.execute
        %e0 = stream.cmd.execute await(%tp0) => with(%t0 as %c: !stream.resource<transient>{%c512}) {
          stream.cmd.fill %c0_i32, %c[%c0 for %c512] : i32 -> !stream.resource<transient>{%c512}
        } => !stream.timepoint
        %dt0 = stream.resource.dealloca await(%e0) => %t0 : !stream.resource<transient>{%c512} => !stream.timepoint
        scf.yield %dt0 : !stream.timepoint
      } else {
        // CHECK: stream.resource.subview %{{.+}}[%[[PACK]]#1]
        %t1, %tp1 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c256} => !stream.timepoint
        // CHECK: stream.cmd.execute
        %e1 = stream.cmd.execute await(%tp1) => with(%t1 as %c: !stream.resource<transient>{%c256}) {
          stream.cmd.fill %c0_i32, %c[%c0 for %c256] : i32 -> !stream.resource<transient>{%c256}
        } => !stream.timepoint
        %dt1 = stream.resource.dealloca await(%e1) => %t1 : !stream.resource<transient>{%c256} => !stream.timepoint
        scf.yield %dt1 : !stream.timepoint
      }
      scf.yield %l2_tp : !stream.timepoint
    } else {
      // Slot 1 subview for middle level else.
      // CHECK: stream.resource.subview %{{.+}}[%[[PACK]]#3]
      %t2, %tp2 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c128} => !stream.timepoint
      // CHECK: stream.cmd.execute
      %e2 = stream.cmd.execute await(%tp2) => with(%t2 as %c: !stream.resource<transient>{%c128}) {
        stream.cmd.fill %c0_i32, %c[%c0 for %c128] : i32 -> !stream.resource<transient>{%c128}
      } => !stream.timepoint
      %dt2 = stream.resource.dealloca await(%e2) => %t2 : !stream.resource<transient>{%c128} => !stream.timepoint
      scf.yield %dt2 : !stream.timepoint
    }
    scf.yield %l1_tp : !stream.timepoint
  } else {
    // Slot 2 subview for outermost else.
    // CHECK: stream.resource.subview %{{.+}}[%[[PACK]]#2]
    %t3, %tp3 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c64} => !stream.timepoint
    // CHECK: stream.cmd.execute
    %e3 = stream.cmd.execute await(%tp3) => with(%t3 as %c: !stream.resource<transient>{%c64}) {
      stream.cmd.fill %c0_i32, %c[%c0 for %c64] : i32 -> !stream.resource<transient>{%c64}
    } => !stream.timepoint
    %dt3 = stream.resource.dealloca await(%e3) => %t3 : !stream.resource<transient>{%c64} => !stream.timepoint
    scf.yield %dt3 : !stream.timepoint
  }

  %result, %result_tp = stream.resource.transients await(%final_tp) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // CHECK-NOT: stream.resource.alloca
  // CHECK-NOT: stream.resource.dealloca
  // CHECK-NOT: stream.resource.transients
  // CHECK: util.return %[[ARG0]], %[[ARG0_SIZE]]
  util.return %result, %arg0_size : !stream.resource<*>, index
}
