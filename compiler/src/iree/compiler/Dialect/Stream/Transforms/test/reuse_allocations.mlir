// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-stream-reuse-allocations))' %s | FileCheck %s

// Tests that a direct reuse of a deallocated resource is reused.

// CHECK-LABEL: @reuseResourceDirect
// CHECK-SAME: (%[[INPUT_TIMEPOINT:.+]]: !stream.timepoint, %[[INPUT_RESOURCE:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
util.func private @reuseResourceDirect(%input_timepoint: !stream.timepoint, %input_resource: !stream.resource<transient>, %size: index) -> (!stream.resource<transient>, !stream.timepoint) {
  // CHECK-NOT: stream.resource.dealloca
  %dealloca_timepoint = stream.resource.dealloca await(%input_timepoint) => %input_resource : !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK-NOT: stream.resource.alloca
  %output_resource, %alloca_timepoint = stream.resource.alloca uninitialized await(%dealloca_timepoint) =>  !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: util.return %[[INPUT_RESOURCE]], %[[INPUT_TIMEPOINT]]
  util.return %output_resource, %alloca_timepoint : !stream.resource<transient>, !stream.timepoint
}

// -----

// Tests that resource types are checked for compatibility before reusing.

// CHECK-LABEL: @reuseTypeMismatch
util.func private @reuseTypeMismatch(%input_timepoint: !stream.timepoint, %input_resource: !stream.resource<transient>, %size: index) -> (!stream.resource<external>, !stream.timepoint) {
  // CHECK: stream.resource.dealloca
  %dealloca_timepoint = stream.resource.dealloca await(%input_timepoint) => %input_resource : !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: stream.resource.alloca
  %output_resource, %alloca_timepoint = stream.resource.alloca uninitialized await(%dealloca_timepoint) =>  !stream.resource<external>{%size} => !stream.timepoint
  util.return %output_resource, %alloca_timepoint : !stream.resource<external>, !stream.timepoint
}

// -----

// Tests that affinities are checked for compatibility before reusing.

// CHECK-LABEL: @reuseAffinityMismatch
util.func private @reuseAffinityMismatch(%input_timepoint: !stream.timepoint, %input_resource: !stream.resource<transient>, %size: index) -> (!stream.resource<transient>, !stream.timepoint) {
  // CHECK: stream.resource.dealloca
  %dealloca_timepoint = stream.resource.dealloca on(#hal.device.promise<@device0>) await(%input_timepoint) => %input_resource : !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: stream.resource.alloca
  %output_resource, %alloca_timepoint = stream.resource.alloca uninitialized on(#hal.device.promise<@device1>) await(%dealloca_timepoint) => !stream.resource<transient>{%size} => !stream.timepoint
  util.return %output_resource, %alloca_timepoint : !stream.resource<transient>, !stream.timepoint
}

// -----

// Tests that only block-local deallocations are checked in the current local
// analysis.

// CHECK-LABEL: @reuseInBlockOnly
// CHECK-SAME: (%[[INPUT_TIMEPOINT:.+]]: !stream.timepoint, %[[INPUT_RESOURCE:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index)
util.func private @reuseInBlockOnly(%input_timepoint: !stream.timepoint, %input_resource: !stream.resource<transient>, %size: index) -> (!stream.resource<transient>, !stream.timepoint) {
  // CHECK: stream.resource.dealloca
  %dealloca_timepoint = stream.resource.dealloca await(%input_timepoint) => %input_resource : !stream.resource<transient>{%size} => !stream.timepoint
  cf.br ^bb2
^bb2:
  // CHECK: stream.resource.alloca
  %output_resource, %alloca_timepoint = stream.resource.alloca uninitialized await(%dealloca_timepoint) =>  !stream.resource<transient>{%size} => !stream.timepoint
  util.return %output_resource, %alloca_timepoint : !stream.resource<transient>, !stream.timepoint
}

// -----

// Tests that the reuse selection logic crosses timeline join ops.

// CHECK-LABEL: @reuseResourceThroughJoin
// CHECK-SAME: (%[[INPUT_TIMEPOINT:.+]]: !stream.timepoint, %[[INPUT_RESOURCE:.+]]: !stream.resource<transient>, %[[SIZE:.+]]: index, %[[OTHER_TIMEPOINT:.+]]: !stream.timepoint)
util.func private @reuseResourceThroughJoin(%input_timepoint: !stream.timepoint, %input_resource: !stream.resource<transient>, %size: index, %other_timepoint: !stream.timepoint) -> (!stream.resource<transient>, !stream.timepoint) {
  // CHECK-NOT: stream.resource.dealloca
  %dealloca_timepoint = stream.resource.dealloca await(%input_timepoint) => %input_resource : !stream.resource<transient>{%size} => !stream.timepoint
  %join_timepoint = stream.timepoint.join max(%dealloca_timepoint, %other_timepoint) => !stream.timepoint
  // CHECK-NOT: stream.resource.alloca
  %output_resource, %alloca_timepoint = stream.resource.alloca uninitialized await(%join_timepoint) =>  !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: util.return %[[INPUT_RESOURCE]], %[[INPUT_TIMEPOINT]]
  util.return %output_resource, %alloca_timepoint : !stream.resource<transient>, !stream.timepoint
}
