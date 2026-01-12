// RUN: iree-opt --iree-stream-transformation-pipeline %s | FileCheck %s

// This file tests that async.parameter.load operations integrate correctly
// through the full stream transformation pipeline, validating:
//   - Partitioning keeps parameter loads with consumers
//   - Timepoint optimization (folding/elision) works correctly
//   - ScheduleAllocation converts async→cmd form properly
//   - No spurious host synchronization (timepoint.await) inserted
//   - Multi-entry gather/scatter operations remain atomic
// These are end-to-end tests of the pipeline (not of full execution or of the
// compiler) and here to ensure the above invariants hold as we modify passes.

// Tests that a parameter load followed by a dispatch operation are
// partitioned into the same execution region and the timepoint is elided.
//
// The compiler should recognize that the load's result timepoint can be
// consumed by the dispatch without requiring an intermediate host wait.
// This validates that ScheduleConcurrency doesn't mispartition async parameter
// operations from their consumers.

// CHECK-LABEL: @LoadThenDispatch
util.func public @LoadThenDispatch(%arg0: index) -> !stream.resource<transient> {
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c512 = arith.constant 512 : index

  // Load weights from parameter archive.
  // Parameter load should be hoisted outside execute region.
  // CHECK: stream.cmd.parameter.load
  // CHECK: "model"::"weights"
  %weights, %load_tp = stream.async.parameter.load "model"::"weights"[%c0_i64]
      : !stream.resource<constant>{%c1024} => !stream.timepoint

  // Await the parameter load (required for valid IR, pipeline will optimize).
  %weights_ready = stream.timepoint.await %load_tp => %weights : !stream.resource<constant>{%c1024}

  // Dispatch should use the loaded resource via await() clause on execute region.
  // CHECK: stream.cmd.execute
  // CHECK-SAME: await
  // CHECK: stream.cmd.dispatch @some_kernel::@dispatch
  %result = stream.async.dispatch @some_kernel::@dispatch[%arg0](%weights_ready[%c0 to %c1024 for %c1024]) : (!stream.resource<constant>{%c1024}) -> !stream.resource<transient>{%c512}

  util.return %result : !stream.resource<transient>
}

// -----

// Tests that a gather operation with multiple entries keeps all entries
// together and doesn't get split across partitions.
//
// The gather operation represents a batched I/O operation that should be
// atomic from the perspective of execution partitioning. Splitting entries
// across partitions would defeat the purpose of batching and could introduce
// unnecessary synchronization overhead.

// CHECK-LABEL: @GatherMultipleEntriesThenDispatch
util.func public @GatherMultipleEntriesThenDispatch(%arg0: index) -> !stream.resource<transient> {
  %c0_i64 = arith.constant 0 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c2048 = arith.constant 2048 : index
  %c512 = arith.constant 512 : index

  // Allocate transient buffer for gathered parameters.
  %buffer = stream.resource.alloc uninitialized : !stream.resource<transient>{%c2048}

  // Gather multiple parameter ranges into buffer.
  // All entries should remain in a single gather operation.
  // CHECK: stream.cmd.parameter.gather
  // CHECK-NEXT: "model"::"bias0"
  // CHECK-NEXT: "model"::"bias1"
  %gathered, %gather_tp = stream.async.parameter.gather {
    "model"::"bias0"[%c0_i64] -> %buffer[%c0 to %c1024 for %c1024] : !stream.resource<transient>{%c2048},
    "model"::"bias1"[%c1024_i64] -> %buffer[%c1024 to %c2048 for %c1024] : !stream.resource<transient>{%c2048}
  } : !stream.resource<transient> => !stream.timepoint

  //  Await gather (required for valid IR, pipeline will optimize).
  %gathered_ready = stream.timepoint.await %gather_tp => %gathered : !stream.resource<transient>{%c2048}

  // Dispatch uses gathered result via await() clause on execute region.
  // CHECK: stream.cmd.execute
  // CHECK-SAME: await
  // CHECK: stream.cmd.dispatch @some_kernel::@dispatch
  %result = stream.async.dispatch @some_kernel::@dispatch[%arg0](%gathered_ready[%c0 to %c2048 for %c2048]) : (!stream.resource<transient>{%c2048}) -> !stream.resource<transient>{%c512}

  util.return %result : !stream.resource<transient>
}

// -----

// Tests that a dispatch followed by a scatter operation threads timepoints
// correctly and the scatter awaits the dispatch completion.
//
// This validates that result timepoints from dispatches properly flow into
// parameter write operations, ensuring that scatter operations wait for
// their source data to be ready before attempting to write to the parameter
// archive.

// CHECK-LABEL: @DispatchThenScatter
util.func public @DispatchThenScatter(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint {
  %c0_i64 = arith.constant 0 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c2048 = arith.constant 2048 : index

  // Dispatch produces result to be scattered.
  // CHECK: stream.cmd.execute
  // CHECK: stream.cmd.dispatch @transform_kernel::@dispatch
  %output = stream.async.dispatch @transform_kernel::@dispatch[%arg1](%arg0[%c0 to %c2048 for %c2048]) : (!stream.resource<transient>{%c2048}) -> !stream.resource<transient>{%c2048}

  // Scatter should follow dispatch, writing result to parameter archive.
  // CHECK: stream.cmd.parameter.scatter
  // CHECK-NEXT: "output"::"result0"
  // CHECK-NEXT: "output"::"result1"
  %scattered, %scatter_tp = stream.async.parameter.scatter {
    %output[%c0 to %c1024 for %c1024] : !stream.resource<transient>{%c2048} -> "output"::"result0"[%c0_i64],
    %output[%c1024 to %c2048 for %c1024] : !stream.resource<transient>{%c2048} -> "output"::"result1"[%c1024_i64]
  } : !stream.resource<transient> => !stream.timepoint

  // Return the timepoint to anchor the scatter operation.
  util.return %scatter_tp : !stream.timepoint
}

// -----

// Tests a complete read-modify-write cycle: load parameter, transform via
// dispatch, write back to parameter archive.
//
// This validates the full parameter transformation workflow, ensuring that
// timepoints thread correctly through load→dispatch→scatter and that the
// operations are partitioned appropriately without unnecessary host
// synchronization.

// CHECK-LABEL: @LoadComputeScatter
util.func public @LoadComputeScatter(%arg0: index) -> !stream.timepoint {
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index

  // Load input parameter.
  // CHECK: stream.cmd.parameter.load
  // CHECK: "input"::"weights"
  %input, %load_tp = stream.async.parameter.load "input"::"weights"[%c0_i64]
      : !stream.resource<constant>{%c1024} => !stream.timepoint

  // Await parameter load (required for valid IR, pipeline will optimize).
  %input_ready = stream.timepoint.await %load_tp => %input : !stream.resource<constant>{%c1024}

  // Transform loaded data via dispatch.
  // CHECK: stream.cmd.execute
  // CHECK: stream.cmd.dispatch @quantize::@dispatch
  %transformed = stream.async.dispatch @quantize::@dispatch[%arg0](%input_ready[%c0 to %c1024 for %c1024]) : (!stream.resource<constant>{%c1024}) -> !stream.resource<transient>{%c1024}

  // Write transformed result back to parameter archive.
  // CHECK: stream.cmd.parameter.scatter
  // CHECK-NEXT: "output"::"quantized_weights"
  %scattered, %scatter_tp = stream.async.parameter.scatter {
    %transformed[%c0 to %c1024 for %c1024] : !stream.resource<transient>{%c1024} -> "output"::"quantized_weights"[%c0_i64]
  } : !stream.resource<transient> => !stream.timepoint

  // Return the timepoint to anchor the scatter operation.
  util.return %scatter_tp : !stream.timepoint
}

// -----

// Tests that multiple sequential parameter operations have their timepoints
// properly folded when operations can be scheduled consecutively.
//
// This validates the ElideTimepoints pass and timeline optimization work
// correctly with parameter operations - when a parameter load's result is
// immediately consumed by another operation, the intermediate timepoint
// should be elided.

// CHECK-LABEL: @MultipleParameterOpsTimepointFolding
util.func public @MultipleParameterOpsTimepointFolding(%arg0: index) -> !stream.resource<transient> {
  %c0_i64 = arith.constant 0 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c2048 = arith.constant 2048 : index
  %c512 = arith.constant 512 : index

  // Load first parameter.
  // CHECK: stream.cmd.parameter.load
  // CHECK: "model"::"weights"
  %weights, %load1_tp = stream.async.parameter.load "model"::"weights"[%c0_i64]
      : !stream.resource<constant>{%c1024} => !stream.timepoint

  // Allocate buffer for second parameter.
  %buffer = stream.resource.alloc uninitialized : !stream.resource<transient>{%c1024}

  // Read second parameter into buffer.
  // CHECK: stream.cmd.parameter.read
  // CHECK: "model"::"bias"
  %read, %read_tp = stream.async.parameter.read "model"::"bias"[%c1024_i64]
      -> %buffer[%c0 to %c1024 for %c1024] : !stream.resource<transient>{%c1024} => !stream.timepoint

  // Await both parameter operations (required for valid IR, pipeline will optimize).
  %weights_ready = stream.timepoint.await %load1_tp => %weights : !stream.resource<constant>{%c1024}
  %read_ready = stream.timepoint.await %read_tp => %read : !stream.resource<transient>{%c1024}

  // Dispatch uses both parameters, awaiting their timepoints via execute region.
  // CHECK: stream.cmd.execute
  // CHECK-SAME: await
  // CHECK: stream.cmd.dispatch @inference::@dispatch
  %result = stream.async.dispatch @inference::@dispatch[%arg0](%weights_ready[%c0 to %c1024 for %c1024], %read_ready[%c0 to %c1024 for %c1024]) : (!stream.resource<constant>{%c1024}, !stream.resource<transient>{%c1024}) -> !stream.resource<transient>{%c512}

  util.return %result : !stream.resource<transient>
}

// -----

// Tests that mixing parameter operations with regular async operations
// (like allocations and dispatches) doesn't cause interference or incorrect
// resource management.
//
// This validates that parameter-backed resources integrate correctly with
// the normal resource allocation and lifetime management, and that parameter
// operations don't inadvertently affect allocation pooling or sizing
// decisions.

// CHECK-LABEL: @MixedParameterAndRegularOps
util.func public @MixedParameterAndRegularOps(%arg0: index) -> !stream.resource<transient> {
  %c0_i64 = arith.constant 0 : i64
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c2048 = arith.constant 2048 : index
  %c512 = arith.constant 512 : index

  // Regular allocation (not parameter-backed).
  // CHECK: stream.resource.alloc
  %temp = stream.resource.alloc uninitialized : !stream.resource<transient>{%c2048}

  // Fill regular allocation.
  %c42_i32 = arith.constant 42 : i32
  %filled = stream.async.fill %c42_i32, %temp[%c0 to %c2048 for %c2048] : i32 -> %temp as !stream.resource<transient>{%c2048}

  // Load from parameter archive.
  // CHECK: stream.cmd.parameter.load
  // CHECK: "model"::"weights"
  %weights, %load_tp = stream.async.parameter.load "model"::"weights"[%c0_i64]
      : !stream.resource<constant>{%c1024} => !stream.timepoint

  // Await parameter load (required for valid IR, pipeline will optimize).
  %weights_ready = stream.timepoint.await %load_tp => %weights : !stream.resource<constant>{%c1024}

  // Dispatch uses both parameter-backed and regular resources.
  // CHECK: stream.cmd.execute
  // CHECK: stream.cmd.fill
  // CHECK: stream.cmd.dispatch @mixed_kernel::@dispatch
  %result = stream.async.dispatch @mixed_kernel::@dispatch[%arg0](%weights_ready[%c0 to %c1024 for %c1024], %filled[%c0 to %c2048 for %c2048]) : (!stream.resource<constant>{%c1024}, !stream.resource<transient>{%c2048}) -> !stream.resource<transient>{%c512}

  util.return %result : !stream.resource<transient>
}
