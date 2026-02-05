// RUN: iree-opt --split-input-file --iree-stream-emplace-transients --verify-diagnostics %s | FileCheck %s

// Tests no transients annotation - pass should no-op.

// CHECK-LABEL: @no_transients
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG0_SIZE:.+]]: index)
util.func public @no_transients(%arg0: !stream.resource<*>, %arg0_size: index) -> (!stream.resource<*>, index) {
  // CHECK-NEXT: util.return %[[ARG0]], %[[ARG0_SIZE]] : !stream.resource<*>, index
  util.return %arg0, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests that we gracefully handle external functions.

// CHECK-LABEL: @external
util.func private @external(%arg0: !stream.resource<*>, %arg0_size: index) -> (!stream.resource<*>, index)

// -----

// Tests zero allocations with transients annotation.
// The pass should remove the transients op (no pack needed) and forward
// both the resource and timepoint.

// CHECK-LABEL: @zero_allocations
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG0_SIZE:.+]]: index, %[[STORAGE:.+]]: !stream.resource<transient>, %[[STORAGE_SIZE:.+]]: index)
util.func public @zero_allocations(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index
) -> (!stream.resource<*>, index) {
  // CHECK-NEXT: arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-NEXT: %[[IMMEDIATE:.+]] = stream.timepoint.immediate
  %immediate = stream.timepoint.immediate => !stream.timepoint

  // Annotate result with transients storage (immediate timepoint - no allocations).
  // The transients op should be removed since there are no allocas.
  // CHECK-NOT: stream.resource.transients
  %result, %result_tp = stream.resource.transients await(%immediate) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // Await should use the forwarded timepoint (was the transients op's await timepoint).
  // CHECK-NEXT: %[[AWAITED:.+]] = stream.timepoint.await %[[IMMEDIATE]] => %[[ARG0]] : !stream.resource<*>{%[[ARG0_SIZE]]}
  %awaited = stream.timepoint.await %result_tp => %result : !stream.resource<*>{%arg0_size}

  // CHECK-NEXT: util.return %[[AWAITED]], %[[ARG0_SIZE]] : !stream.resource<*>, index
  util.return %awaited, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests a single allocation with constant size.

// CHECK-LABEL: @single_allocation_constant
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<*>, %[[ARG0_SIZE:.+]]: index, %[[STORAGE:.+]]: !stream.resource<transient>, %[[STORAGE_SIZE:.+]]: index, %[[INPUT_TIMEPOINT:.+]]: !stream.timepoint)
util.func public @single_allocation_constant(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index,
  %input_timepoint: !stream.timepoint
) -> (!stream.resource<*>, index) {
  // CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: %[[TRANSIENT_SIZE:.+]] = arith.constant 1024 : index
  %c1024 = arith.constant 1024 : index

  // Size hoisted for pack (shows transformation happening).
  // CHECK: %[[TRANSIENT_SIZE_FOR_PACK:.+]] = arith.constant 1024 : index
  //
  // Pack created with single slice - total size in #0, offset in #1.
  // CHECK-NEXT: %[[PACK:.+]]:2 = stream.resource.pack slices({
  // CHECK-NEXT:   [0, 0] = %[[TRANSIENT_SIZE_FOR_PACK]]
  // CHECK-NEXT: }) : index attributes {stream.experimental.transients}
  //
  // Storage subview at offset 0.
  // CHECK: %[[BASE_OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[STORAGE_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE]][%[[BASE_OFFSET]]] : !stream.resource<transient>{%[[STORAGE_SIZE]]} -> !stream.resource<transient>{%[[PACK]]#0}
  //
  // Allocation replaced with subview from storage (uses pack offset #1).
  // CHECK: %[[ALLOC_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_SUBVIEW]][%[[PACK]]#1] : !stream.resource<transient>{%[[PACK]]#0} -> !stream.resource<transient>{%[[TRANSIENT_SIZE]]}

  // Allocate transient resource awaiting input timepoint.
  %transient, %alloca_timepoint = stream.resource.alloca uninitialized await(%input_timepoint) =>
      !stream.resource<transient>{%c1024} => !stream.timepoint

  // CRITICAL: Input timepoint threads through to execute (alloca removed, timeline preserved).
  // CHECK: %[[EXEC_TP:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[ALLOC_SUBVIEW]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[TRANSIENT_SIZE]]}) {
  // CHECK-NEXT:   stream.cmd.fill %[[ZERO_I32]], %[[CAPTURE]][%[[ZERO_INDEX]] for %[[TRANSIENT_SIZE]]] : i32 -> !stream.resource<transient>{%[[TRANSIENT_SIZE]]}
  // CHECK-NEXT: } => !stream.timepoint
  %exec_timepoint = stream.cmd.execute
      await(%alloca_timepoint) => with(%transient as %transient_inner: !stream.resource<transient>{%c1024}) {
    stream.cmd.fill %c0_i32, %transient_inner[%c0 for %c1024] : i32 -> !stream.resource<transient>{%c1024}
  } => !stream.timepoint

  // Deallocate transient.
  %dealloca_timepoint = stream.resource.dealloca await(%exec_timepoint) => %transient : !stream.resource<transient>{%c1024} => !stream.timepoint

  // Annotate result with transients storage, threading the timepoint.
  %result_annotated, %result_tp = stream.resource.transients await(%dealloca_timepoint) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // Verify transformations occurred.
  // CHECK-NOT: stream.resource.alloca
  // CHECK-NOT: stream.resource.dealloca
  // CHECK-NOT: stream.resource.transients
  //
  // CHECK: util.return %[[ARG0]], %[[ARG0_SIZE]]
  util.return %result_annotated, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests a single allocation with a dynamic size.

// CHECK-LABEL: @single_allocation_dynamic
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>, %[[STORAGE_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[INPUT_TIMEPOINT:[a-z0-9]+]]: !stream.timepoint)
util.func public @single_allocation_dynamic(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index,
  %input_timepoint: !stream.timepoint
) -> (!stream.resource<*>, index) {
  // CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: %[[BYTES_PER_ELEMENT:.+]] = arith.constant 4 : index
  %c4 = arith.constant 4 : index

  // Original size computation (preserved for execution).
  // CHECK: %[[TRANSIENT_SIZE:.+]] = arith.muli %[[ARG0_SIZE]], %[[BYTES_PER_ELEMENT]] : index
  %transient_size = arith.muli %arg0_size, %c4 : index

  // Hoisted size computation for pack (cloned).
  // CHECK: %[[BYTES_PER_ELEMENT_CLONE:.+]] = arith.constant 4 : index
  // CHECK-NEXT: %[[TRANSIENT_SIZE_FOR_PACK:.+]] = arith.muli %[[ARG0_SIZE]], %[[BYTES_PER_ELEMENT_CLONE]] : index
  //
  // Pack created with hoisted size.
  // CHECK: %[[PACK:.+]]:2 = stream.resource.pack slices({
  // CHECK-NEXT:   [0, 0] = %[[TRANSIENT_SIZE_FOR_PACK]]
  // CHECK-NEXT: }) : index attributes {stream.experimental.transients}
  //
  // Storage subview at offset 0.
  // CHECK: %[[BASE_OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[STORAGE_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE]][%[[BASE_OFFSET]]] : !stream.resource<transient>{%[[STORAGE_SIZE]]} -> !stream.resource<transient>{%[[PACK]]#0}
  //
  // Allocation replaced with subview (uses pack offset #1).
  // CHECK: %[[ALLOC_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_SUBVIEW]][%[[PACK]]#1] : !stream.resource<transient>{%[[PACK]]#0} -> !stream.resource<transient>{%[[TRANSIENT_SIZE]]}

  // Allocate transient resource awaiting input timepoint.
  %transient, %alloca_timepoint = stream.resource.alloca uninitialized await(%input_timepoint) =>
      !stream.resource<transient>{%transient_size} => !stream.timepoint

  // CRITICAL: Input timepoint threads through to execute (alloca removed, timeline preserved).
  // CHECK: %[[EXEC_TP:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[ALLOC_SUBVIEW]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[TRANSIENT_SIZE]]}) {
  // CHECK-NEXT:   stream.cmd.fill %[[ZERO_I32]], %[[CAPTURE]][%[[ZERO_INDEX]] for %[[TRANSIENT_SIZE]]] : i32 -> !stream.resource<transient>{%[[TRANSIENT_SIZE]]}
  // CHECK-NEXT: } => !stream.timepoint
  %exec_timepoint = stream.cmd.execute
      await(%alloca_timepoint) => with(%transient as %transient_inner: !stream.resource<transient>{%transient_size}) {
    stream.cmd.fill %c0_i32, %transient_inner[%c0 for %transient_size] : i32 -> !stream.resource<transient>{%transient_size}
  } => !stream.timepoint

  // Deallocate transient.
  %dealloca_timepoint = stream.resource.dealloca await(%exec_timepoint) => %transient : !stream.resource<transient>{%transient_size} => !stream.timepoint

  // Thread timepoint to transients op.
  %result_annotated, %result_tp = stream.resource.transients await(%dealloca_timepoint) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // Verify transformations occurred.
  // CHECK-NOT: stream.resource.alloca
  // CHECK-NOT: stream.resource.dealloca
  // CHECK-NOT: stream.resource.transients
  //
  // CHECK: util.return %[[ARG0]], %[[ARG0_SIZE]]
  util.return %result_annotated, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests multiple allocations (non-overlapping liveness).

// CHECK-LABEL: @two_allocations
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>, %[[STORAGE_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[INPUT_TIMEPOINT:[a-z0-9]+]]: !stream.timepoint)
util.func public @two_allocations(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index,
  %input_timepoint: !stream.timepoint
) -> (!stream.resource<*>, index) {
  // CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: %[[FIRST_BUFFER_SIZE:.+]] = arith.constant 512 : index
  %c512 = arith.constant 512 : index
  // CHECK-DAG: %[[SECOND_BUFFER_SIZE:.+]] = arith.constant 1024 : index
  %c1024 = arith.constant 1024 : index
  // CHECK-DAG: %[[ONE_I32:.+]] = arith.constant 1 : i32
  %c1_i32 = arith.constant 1 : i32

  // Sizes hoisted for pack in program order.
  // CHECK: %[[FIRST_BUFFER_SIZE_FOR_PACK:.+]] = arith.constant 512 : index
  // CHECK-NEXT: %[[SECOND_BUFFER_SIZE_FOR_PACK:.+]] = arith.constant 1024 : index
  //
  // Pack with two slices in program order - both [0,0] means non-overlapping liveness.
  // CHECK-NEXT: %[[PACK:.+]]:3 = stream.resource.pack slices({
  // CHECK-NEXT:   [0, 0] = %[[FIRST_BUFFER_SIZE_FOR_PACK]],
  // CHECK-NEXT:   [0, 0] = %[[SECOND_BUFFER_SIZE_FOR_PACK]]
  // CHECK-NEXT: }) : index attributes {stream.experimental.transients}
  //
  // Storage subview at offset 0.
  // CHECK: %[[BASE_OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[STORAGE_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE]][%[[BASE_OFFSET]]] : !stream.resource<transient>{%[[STORAGE_SIZE]]} -> !stream.resource<transient>{%[[PACK]]#0}

  // First allocation awaiting input timepoint.
  %transient0, %alloca_timepoint0 = stream.resource.alloca uninitialized await(%input_timepoint) =>
      !stream.resource<transient>{%c512} => !stream.timepoint

  // CRITICAL: First allocation execution uses input timepoint (alloca removed, timeline preserved).
  // CRITICAL: First allocation (512) gets pack result #1, subview created where alloca was.
  // CHECK: %[[ALLOC0_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_SUBVIEW]][%[[PACK]]#1] : !stream.resource<transient>{%[[PACK]]#0} -> !stream.resource<transient>{%[[FIRST_BUFFER_SIZE]]}
  // CHECK: %[[EXEC0_TP:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[ALLOC0_SUBVIEW]] as %[[CAPTURE0:.+]]: !stream.resource<transient>{%[[FIRST_BUFFER_SIZE]]}) {
  // CHECK-NEXT:   stream.cmd.fill %[[ZERO_I32]], %[[CAPTURE0]][%[[ZERO_INDEX]] for %[[FIRST_BUFFER_SIZE]]] : i32 -> !stream.resource<transient>{%[[FIRST_BUFFER_SIZE]]}
  // CHECK: } => !stream.timepoint
  %exec_timepoint0 = stream.cmd.execute
      await(%alloca_timepoint0) => with(%transient0 as %t0: !stream.resource<transient>{%c512}) {
    stream.cmd.fill %c0_i32, %t0[%c0 for %c512] : i32 -> !stream.resource<transient>{%c512}
  } => !stream.timepoint

  %dealloca_timepoint0 = stream.resource.dealloca await(%exec_timepoint0) => %transient0 : !stream.resource<transient>{%c512} => !stream.timepoint

  // Second allocation (after first is deallocated) awaiting dealloca timepoint.
  %transient1, %alloca_timepoint1 = stream.resource.alloca uninitialized await(%dealloca_timepoint0) =>
      !stream.resource<transient>{%c1024} => !stream.timepoint

  // CRITICAL: Second allocation (1024) gets pack result #2, subview created where alloca was.
  // CHECK: %[[ALLOC1_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_SUBVIEW]][%[[PACK]]#2] : !stream.resource<transient>{%[[PACK]]#0} -> !stream.resource<transient>{%[[SECOND_BUFFER_SIZE]]}
  // Second allocation execution awaits first dealloca (timeline threading).
  // CHECK: %[[EXEC1_TP:.+]] = stream.cmd.execute await(%[[EXEC0_TP]]) => with(%[[ALLOC1_SUBVIEW]] as %[[CAPTURE1:.+]]: !stream.resource<transient>{%[[SECOND_BUFFER_SIZE]]}) {
  // CHECK-NEXT:   stream.cmd.fill %[[ONE_I32]], %[[CAPTURE1]][%[[ZERO_INDEX]] for %[[SECOND_BUFFER_SIZE]]] : i32 -> !stream.resource<transient>{%[[SECOND_BUFFER_SIZE]]}
  // CHECK-NEXT: } => !stream.timepoint
  %exec_timepoint1 = stream.cmd.execute
      await(%alloca_timepoint1) => with(%transient1 as %t1: !stream.resource<transient>{%c1024}) {
    stream.cmd.fill %c1_i32, %t1[%c0 for %c1024] : i32 -> !stream.resource<transient>{%c1024}
  } => !stream.timepoint

  %dealloca_timepoint1 = stream.resource.dealloca await(%exec_timepoint1) => %transient1 : !stream.resource<transient>{%c1024} => !stream.timepoint

  // Timeline join preserved.
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[EXEC0_TP]], %[[EXEC1_TP]]) => !stream.timepoint
  %join = stream.timepoint.join max(%dealloca_timepoint0, %dealloca_timepoint1) => !stream.timepoint

  %result_annotated, %result_tp = stream.resource.transients await(%join) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // Verify transformations occurred.
  // CHECK-NOT: stream.resource.alloca
  // CHECK-NOT: stream.resource.dealloca
  // CHECK-NOT: stream.resource.transients
  //
  // CHECK: util.return %[[ARG0]], %[[ARG0_SIZE]]
  util.return %result_annotated, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests many allocations - should work (just not pack efficiently).

// CHECK-LABEL: @many_allocations
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>, %[[STORAGE_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[INPUT_TIMEPOINT:[a-z0-9]+]]: !stream.timepoint)
util.func public @many_allocations(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index,
  %input_timepoint: !stream.timepoint
) -> (!stream.resource<*>, index) {
  // CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: %[[ALLOCATION_SIZE:.+]] = arith.constant 256 : index
  %c256 = arith.constant 256 : index

  // Sizes hoisted for pack (3 allocations of 256 each).
  // CHECK: %[[ALLOCATION_SIZE_FOR_PACK0:.+]] = arith.constant 256 : index
  // CHECK-NEXT: %[[ALLOCATION_SIZE_FOR_PACK1:.+]] = arith.constant 256 : index
  // CHECK-NEXT: %[[ALLOCATION_SIZE_FOR_PACK2:.+]] = arith.constant 256 : index
  //
  // Pack with three slices - all [0,0] means non-overlapping.
  // CHECK-NEXT: %[[PACK:.+]]:4 = stream.resource.pack slices({
  // CHECK-NEXT:   [0, 0] = %[[ALLOCATION_SIZE_FOR_PACK0]],
  // CHECK-NEXT:   [0, 0] = %[[ALLOCATION_SIZE_FOR_PACK1]],
  // CHECK-NEXT:   [0, 0] = %[[ALLOCATION_SIZE_FOR_PACK2]]
  // CHECK-NEXT: }) : index attributes {stream.experimental.transients}
  //
  // Storage subview.
  // CHECK: %[[BASE_OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[STORAGE_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE]][%[[BASE_OFFSET]]] : !stream.resource<transient>{%[[STORAGE_SIZE]]} -> !stream.resource<transient>{%[[PACK]]#0}

  %t0, %tp0 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c256} => !stream.timepoint
  // Subview created where alloca was (interleaved with executes in program order).
  // CHECK: %[[ALLOC0_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_SUBVIEW]][%[[PACK]]#1]
  // CRITICAL: Input timepoint threads through all executions.
  // CHECK: %[[EXEC0:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[ALLOC0_SUBVIEW]]
  %e0 = stream.cmd.execute await(%tp0) => with(%t0 as %c0_r: !stream.resource<transient>{%c256}) {
    stream.cmd.fill %c0_i32, %c0_r[%c0 for %c256] : i32 -> !stream.resource<transient>{%c256}
  } => !stream.timepoint
  %dt0 = stream.resource.dealloca await(%e0) => %t0 : !stream.resource<transient>{%c256} => !stream.timepoint

  %t1, %tp1 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c256} => !stream.timepoint
  // CHECK: %[[ALLOC1_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_SUBVIEW]][%[[PACK]]#2]
  // CHECK: %[[EXEC1:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[ALLOC1_SUBVIEW]]
  %e1 = stream.cmd.execute await(%tp1) => with(%t1 as %c1_r: !stream.resource<transient>{%c256}) {
    stream.cmd.fill %c0_i32, %c1_r[%c0 for %c256] : i32 -> !stream.resource<transient>{%c256}
  } => !stream.timepoint
  %dt1 = stream.resource.dealloca await(%e1) => %t1 : !stream.resource<transient>{%c256} => !stream.timepoint

  %t2, %tp2 = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%c256} => !stream.timepoint
  // CHECK: %[[ALLOC2_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_SUBVIEW]][%[[PACK]]#3]
  // CHECK: %[[EXEC2:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[ALLOC2_SUBVIEW]]
  %e2 = stream.cmd.execute await(%tp2) => with(%t2 as %c2_r: !stream.resource<transient>{%c256}) {
    stream.cmd.fill %c0_i32, %c2_r[%c0 for %c256] : i32 -> !stream.resource<transient>{%c256}
  } => !stream.timepoint
  %dt2 = stream.resource.dealloca await(%e2) => %t2 : !stream.resource<transient>{%c256} => !stream.timepoint

  // Three-way join preserved (order may vary).
  // CHECK: %[[JOIN:.+]] = stream.timepoint.join max(%[[EXEC0]], %[[EXEC1]], %[[EXEC2]]) => !stream.timepoint
  %join = stream.timepoint.join max(%dt0, %dt1, %dt2) => !stream.timepoint
  %result, %result_tp = stream.resource.transients await(%join) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // CHECK-NOT: stream.resource.alloca
  // CHECK-NOT: stream.resource.dealloca
  // CHECK-NOT: stream.resource.transients
  // CHECK: util.return %[[ARG0]], %[[ARG0_SIZE]]
  util.return %result, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests that functions which call another function fail (today).

util.func private @helper(%arg0: !stream.resource<*>, %arg0_size: index) -> (!stream.resource<*>, index) {
  util.return %arg0, %arg0_size : !stream.resource<*>, index
}

// expected-error @+1 {{function contains function calls}}
util.func public @has_function_call(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index
) -> (!stream.resource<*>, index) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c512 = arith.constant 512 : index

  %t0, %tp0 = stream.resource.alloca uninitialized : !stream.resource<transient>{%c512} => !stream.timepoint
  %e0 = stream.cmd.execute await(%tp0) => with(%t0 as %c: !stream.resource<transient>{%c512}) {
    stream.cmd.fill %c0_i32, %c[%c0 for %c512] : i32 -> !stream.resource<transient>{%c512}
  } => !stream.timepoint
  %dt0 = stream.resource.dealloca await(%e0) => %t0 : !stream.resource<transient>{%c512} => !stream.timepoint

  %result, %result_size = util.call @helper(%arg0, %arg0_size) : (!stream.resource<*>, index) -> (!stream.resource<*>, index)
  %result_annotated, %result_tp = stream.resource.transients await(%dt0) => %result : !stream.resource<*>{%result_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  util.return %result_annotated, %result_size : !stream.resource<*>, index
}

// -----

// Tests private function with transients fail (only public functions supported
// today).

// expected-error @+1 {{only public functions}}
util.func private @private_with_transients(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index
) -> (!stream.resource<*>, index) {
  %immediate = stream.timepoint.immediate => !stream.timepoint
  %result_annotated, %result_tp = stream.resource.transients await(%immediate) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint
  util.return %result_annotated, %arg0_size : !stream.resource<*>, index
}

// -----

// Tests that deallocas for non-emplaced transients are NOT removed.
// This verifies we only remove deallocas for the specific allocas we're emplacing.

// CHECK-LABEL: @dealloca_filtering
// CHECK-SAME: (%[[ARG0:.+]]: !stream.resource<transient>, %[[ARG0_SIZE:.+]]: index, %[[STORAGE:.+]]: !stream.resource<transient>, %[[STORAGE_SIZE:.+]]: index)
util.func public @dealloca_filtering(
  %arg0: !stream.resource<transient>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index
) -> (!stream.resource<transient>, index) {
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index

  // This alloca should be emplaced (removed).
  // CHECK: stream.resource.pack
  // CHECK: stream.resource.subview %[[STORAGE]]
  // CHECK: stream.resource.subview
  %transient, %alloca_tp = stream.resource.alloca uninitialized : !stream.resource<transient>{%c512} => !stream.timepoint

  // CHECK-NEXT: stream.timepoint.immediate
  // Dealloca for %arg0 (not emplaced) - will be preserved.
  // CHECK-NEXT: %[[DEALLOCA:.+]] = stream.resource.dealloca await(%[[ALLOCA_TP:.+]]) => %[[ARG0]] : !stream.resource<transient>{%[[ARG0_SIZE]]}
  %dealloca_arg0_tp = stream.resource.dealloca await(%alloca_tp) => %arg0 : !stream.resource<transient>{%arg0_size} => !stream.timepoint

  // Dealloca for %transient (emplaced) - will be removed.
  %dealloca_transient_tp = stream.resource.dealloca await(%dealloca_arg0_tp) => %transient : !stream.resource<transient>{%c512} => !stream.timepoint

  %result, %result_tp = stream.resource.transients await(%dealloca_transient_tp) => %arg0 : !stream.resource<transient>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // CHECK-NEXT: %[[AWAIT_RESULT:.+]] = stream.timepoint.await %[[DEALLOCA]] => %[[ARG0]]
  %final = stream.timepoint.await %result_tp => %result : !stream.resource<transient>{%arg0_size}
  // CHECK-NEXT: util.return
  util.return %final, %arg0_size : !stream.resource<transient>, index
}

// -----

// TODO(benvanik): Add test for host synchronization detection once implemented.

// -----

// Tests that size computation with arith ops is hoisted to function entry.

// CHECK-LABEL: @size_with_arith_ops
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: !stream.resource<*>, %[[ARG0_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[STORAGE:[a-z0-9]+]]: !stream.resource<transient>, %[[STORAGE_SIZE:[a-z0-9]+]]: index,
// CHECK-SAME:  %[[INPUT_TIMEPOINT:[a-z0-9]+]]: !stream.timepoint)
util.func public @size_with_arith_ops(
  %arg0: !stream.resource<*>, %arg0_size: index,
  %storage: !stream.resource<transient>, %storage_size: index,
  %input_timepoint: !stream.timepoint
) -> (!stream.resource<*>, index) {
  // CHECK-DAG: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[ZERO_I32:.+]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: %[[BYTES_PER_ELEMENT:.+]] = arith.constant 4 : index
  %c4 = arith.constant 4 : index
  // CHECK-DAG: %[[BASE_PADDING:.+]] = arith.constant 1024 : index
  %c1024 = arith.constant 1024 : index

  // Original size computation (preserved for execution).
  // CHECK: %[[BASE_SIZE:.+]] = arith.muli %[[ARG0_SIZE]], %[[BYTES_PER_ELEMENT]] : index
  %base_size = arith.muli %arg0_size, %c4 : index
  // CHECK: %[[TRANSIENT_SIZE:.+]] = arith.addi %[[BASE_SIZE]], %[[BASE_PADDING]] : index
  %transient_size = arith.addi %base_size, %c1024 : index

  // Hoisted size computation for pack (cloned).
  // CHECK: %[[BYTES_PER_ELEMENT_CLONE:.+]] = arith.constant 4 : index
  // CHECK-NEXT: %[[BASE_SIZE_FOR_PACK:.+]] = arith.muli %[[ARG0_SIZE]], %[[BYTES_PER_ELEMENT_CLONE]] : index
  // CHECK-NEXT: %[[BASE_PADDING_CLONE:.+]] = arith.constant 1024 : index
  // CHECK-NEXT: %[[TRANSIENT_SIZE_FOR_PACK:.+]] = arith.addi %[[BASE_SIZE_FOR_PACK]], %[[BASE_PADDING_CLONE]] : index
  //
  // Pack uses hoisted computation.
  // CHECK-NEXT: %[[PACK:.+]]:2 = stream.resource.pack slices({
  // CHECK-NEXT:   [0, 0] = %[[TRANSIENT_SIZE_FOR_PACK]]
  // CHECK-NEXT: }) : index attributes {stream.experimental.transients}
  //
  // Storage and allocation subviews.
  // CHECK: %[[BASE_OFFSET:.+]] = arith.constant 0 : index
  // CHECK: %[[STORAGE_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE]][%[[BASE_OFFSET]]] : !stream.resource<transient>{%[[STORAGE_SIZE]]} -> !stream.resource<transient>{%[[PACK]]#0}
  // CHECK: %[[ALLOC_SUBVIEW:.+]] = stream.resource.subview %[[STORAGE_SUBVIEW]][%[[PACK]]#1] : !stream.resource<transient>{%[[PACK]]#0} -> !stream.resource<transient>{%[[TRANSIENT_SIZE]]}

  %transient, %alloca_timepoint = stream.resource.alloca uninitialized await(%input_timepoint) =>
      !stream.resource<transient>{%transient_size} => !stream.timepoint

  // CRITICAL: Execute uses original size computation and input timepoint (alloca removed, timeline preserved).
  // CHECK: %[[EXEC_TP:.+]] = stream.cmd.execute await(%[[INPUT_TIMEPOINT]]) => with(%[[ALLOC_SUBVIEW]] as %[[CAPTURE:.+]]: !stream.resource<transient>{%[[TRANSIENT_SIZE]]}) {
  // CHECK-NEXT:   stream.cmd.fill %[[ZERO_I32]], %[[CAPTURE]][%[[ZERO_INDEX]] for %[[TRANSIENT_SIZE]]] : i32 -> !stream.resource<transient>{%[[TRANSIENT_SIZE]]}
  // CHECK: } => !stream.timepoint
  %exec_timepoint = stream.cmd.execute
      await(%alloca_timepoint) => with(%transient as %t: !stream.resource<transient>{%transient_size}) {
    stream.cmd.fill %c0_i32, %t[%c0 for %transient_size] : i32 -> !stream.resource<transient>{%transient_size}
  } => !stream.timepoint

  %dealloca_timepoint = stream.resource.dealloca await(%exec_timepoint) => %transient : !stream.resource<transient>{%transient_size} => !stream.timepoint

  %result_annotated, %result_tp = stream.resource.transients await(%dealloca_timepoint) => %arg0 : !stream.resource<*>{%arg0_size}
      from %storage : !stream.resource<transient>{%storage_size}
      => !stream.timepoint

  // CHECK-NOT: stream.resource.alloca
  // CHECK-NOT: stream.resource.dealloca
  // CHECK-NOT: stream.resource.transients
  // CHECK: util.return %[[ARG0]], %[[ARG0_SIZE]]
  util.return %result_annotated, %arg0_size : !stream.resource<*>, index
}
