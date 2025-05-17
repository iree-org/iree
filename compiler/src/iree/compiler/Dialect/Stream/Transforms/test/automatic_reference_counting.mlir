// RUN: iree-opt --split-input-file --iree-stream-automatic-reference-counting %s | FileCheck %s

// Tests insertion of deallocations when there are no users.

// CHECK-LABEL: @insertDeallocaNoUses
// CHECK-SAME: (%[[INPUT_TIMEPOINT:.+]]: !stream.timepoint, %[[SIZE:.+]]: index)
util.func private @insertDeallocaNoUses(%input_timepoint: !stream.timepoint, %size: index) -> !stream.timepoint {
  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca
  %resource, %alloca_timepoint = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: %[[DEALLOCA_TIMEPOINT:.+]] = stream.resource.dealloca origin await(%[[ALLOCA_TIMEPOINT]]) => %[[RESOURCE]] : !stream.resource<transient>{%[[SIZE]]} => !stream.timepoint
  // CHECK: util.return %[[DEALLOCA_TIMEPOINT]]
  util.return %alloca_timepoint : !stream.timepoint
}

// -----

// Tests that deallocations have affinities assigned when available.

// CHECK-LABEL: @insertDeallocaWithAffinity
util.func private @insertDeallocaWithAffinity(%input_timepoint: !stream.timepoint, %size: index) -> !stream.timepoint {
  %resource, %alloca_timepoint = stream.resource.alloca uninitialized on(#hal.device.promise<@device>) await(%input_timepoint) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: stream.resource.dealloca on(#hal.device.promise<@device>)
  util.return %alloca_timepoint : !stream.timepoint
}

// -----

// Tests insertion of deallocations when there is one user.

// CHECK-LABEL: @insertDeallocaOneUserOneUse
util.func private @insertDeallocaOneUserOneUse(%input_timepoint: !stream.timepoint, %size: index) -> !stream.timepoint {
  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca
  %resource, %alloca_timepoint = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: %[[EXECUTE_TIMEPOINT:.+]] = stream.cmd.execute
  %execute_timepoint = stream.cmd.execute await(%alloca_timepoint) => with(%resource as %capture : !stream.resource<transient>{%size}) {
  } => !stream.timepoint
  // CHECK: %[[DEALLOCA_TIMEPOINT:.+]] = stream.resource.dealloca origin await(%[[EXECUTE_TIMEPOINT]]) => %[[RESOURCE]]
  // CHECK: util.return %[[DEALLOCA_TIMEPOINT]]
  util.return %execute_timepoint : !stream.timepoint
}

// -----

// Tests insertion of deallocations when there is one user with multiple uses.
// Finds issues with code using hasOneUse - this should behave the same as
// @insertDeallocaOneUserOneUse.

// CHECK-LABEL: @insertDeallocaOneUserMultiUse
util.func private @insertDeallocaOneUserMultiUse(%input_timepoint: !stream.timepoint, %size: index) -> !stream.timepoint {
  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca
  %resource, %alloca_timepoint = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: %[[EXECUTE_TIMEPOINT:.+]] = stream.cmd.execute
  %execute_timepoint = stream.cmd.execute await(%alloca_timepoint) => with(%resource as %capture0 : !stream.resource<transient>{%size}, %resource as %capture1 : !stream.resource<transient>{%size}) {
  } => !stream.timepoint
  // CHECK: %[[DEALLOCA_TIMEPOINT:.+]] = stream.resource.dealloca origin await(%[[EXECUTE_TIMEPOINT]]) => %[[RESOURCE]]
  // CHECK-NOT: stream.resource.dealloca
  // CHECK: util.return %[[DEALLOCA_TIMEPOINT]]
  util.return %execute_timepoint : !stream.timepoint
}

// -----

// Tests insertion of deallocations when there is multiple users.
// These users run sequentially and the last timepoint is used for the timeline.

// CHECK-LABEL: @insertDeallocaMultiUserSequence
util.func private @insertDeallocaMultiUserSequence(%input_timepoint: !stream.timepoint, %size: index) -> !stream.timepoint {
  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca
  %resource, %alloca_timepoint = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: %[[EXECUTE0_TIMEPOINT:.+]] = stream.cmd.execute
  %execute0_timepoint = stream.cmd.execute await(%alloca_timepoint) => with(%resource as %capture : !stream.resource<transient>{%size}) {
  } => !stream.timepoint
  // CHECK: %[[EXECUTE1_TIMEPOINT:.+]] = stream.cmd.execute
  %execute1_timepoint = stream.cmd.execute await(%execute0_timepoint) => with(%resource as %capture : !stream.resource<transient>{%size}) {
  } => !stream.timepoint
  // CHECK: %[[DEALLOCA_TIMEPOINT:.+]] = stream.resource.dealloca origin await(%[[EXECUTE1_TIMEPOINT]]) => %[[RESOURCE]]
  // Note: needs cleanup in ElideTimepointsPass.
  // CHECK: %[[EXECUTE_JOIN_TIMEPOINT:.+]] = stream.timepoint.join max(%[[EXECUTE0_TIMEPOINT]], %[[DEALLOCA_TIMEPOINT]])
  %execute_join_timepoint = stream.timepoint.join max(%execute0_timepoint, %execute1_timepoint) => !stream.timepoint
  // CHECK: util.return %[[EXECUTE_JOIN_TIMEPOINT]]
  util.return %execute_join_timepoint : !stream.timepoint
}

// -----

// Tests insertion of deallocations when there is multiple users.
// These users fork and need a join inserted.

// CHECK-LABEL: @insertDeallocaMultiUserFork
util.func private @insertDeallocaMultiUserFork(%input_timepoint: !stream.timepoint, %size: index) -> (!stream.timepoint, !stream.timepoint) {
  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca
  %resource, %alloca_timepoint = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: %[[EXECUTE0_TIMEPOINT:.+]] = stream.cmd.execute await(%[[ALLOCA_TIMEPOINT]])
  %execute0_timepoint = stream.cmd.execute await(%alloca_timepoint) => with(%resource as %capture : !stream.resource<transient>{%size}) {
  } => !stream.timepoint
  // Note: this is here to force another timepoint user earlier than the last
  // deallocation; this exposes potential SSA ordering issues.
  // CHECK: %[[OTHER_TIMEPOINT:.+]] = stream.cmd.execute await(%[[EXECUTE0_TIMEPOINT]])
  %other_timepoint = stream.cmd.execute await(%execute0_timepoint) => with() {
  } => !stream.timepoint
  // CHECK: %[[EXECUTE1_TIMEPOINT:.+]] = stream.cmd.execute await(%[[ALLOCA_TIMEPOINT]])
  %execute1_timepoint = stream.cmd.execute await(%alloca_timepoint) => with(%resource as %capture : !stream.resource<transient>{%size}) {
  } => !stream.timepoint
  // CHECK: %[[DEALLOCA_JOIN_TIMEPOINT:.+]] = stream.timepoint.join max(%[[EXECUTE0_TIMEPOINT]], %[[EXECUTE1_TIMEPOINT]])
  // CHECK: %[[DEALLOCA_TIMEPOINT:.+]] = stream.resource.dealloca origin await(%[[DEALLOCA_JOIN_TIMEPOINT]]) => %[[RESOURCE]]
  // Note: the dealloca adds an additional synchronization point.
  // CHECK: util.return %[[OTHER_TIMEPOINT]], %[[DEALLOCA_TIMEPOINT]]
  util.return %other_timepoint, %execute1_timepoint : !stream.timepoint, !stream.timepoint
}

// -----

// Tests that deallocations are not inserted when one is already present for
// a particular resource.

// CHECK-LABEL: @ignoreHandledResources
util.func private @ignoreHandledResources(%input_timepoint: !stream.timepoint, %size: index) -> !stream.timepoint {
  // CHECK: stream.resource.alloca
  %resource, %alloca_timepoint = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: stream.cmd.execute
  %execute_timepoint = stream.cmd.execute await(%alloca_timepoint) => with(%resource as %capture : !stream.resource<transient>{%size}) {
  } => !stream.timepoint
  // CHECK: %[[DEALLOCA_TIMEPOINT:.+]] = stream.resource.dealloca
  // CHECK-NOT: stream.resource.dealloca
  %dealloca_timepoint = stream.resource.dealloca await(%execute_timepoint) => %resource : !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: util.return %[[DEALLOCA_TIMEPOINT]]
  util.return %dealloca_timepoint : !stream.timepoint
}

// -----

// Tests that live-in resources are ignored.

// CHECK-LABEL: @ignoreLiveIn
util.func private @ignoreLiveIn(%input_timepoint: !stream.timepoint, %resource: !stream.resource<transient>, %size: index) -> !stream.timepoint {
  %execute_timepoint = stream.cmd.execute await(%input_timepoint) => with(%resource as %capture : !stream.resource<transient>{%size}) {
  } => !stream.timepoint
  // CHECK-NOT: stream.resource.dealloca
  util.return %execute_timepoint : !stream.timepoint
}

// -----

// Tests that live-out resources are ignored.

// CHECK-LABEL: @ignoreLiveOut
util.func private @ignoreLiveOut(%input_timepoint: !stream.timepoint, %size: index) -> (!stream.resource<transient>, !stream.timepoint) {
  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca
  %resource, %alloca_timepoint = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK-NOT: stream.resource.dealloca
  // CHECK: util.return %[[RESOURCE]], %[[ALLOCA_TIMEPOINT]]
  util.return %resource, %alloca_timepoint : !stream.resource<transient>, !stream.timepoint
}

// -----

// Tests that resources with indeterminate lifetimes tied through other ops
// are not deallocated.

// CHECK-LABEL: @ignoreTiedIndeterminate
util.func private @ignoreTiedIndeterminate(%input_timepoint: !stream.timepoint, %resource: !stream.resource<transient>, %size: index) -> !stream.timepoint {
  // CHECK: stream.timepoint.barrier
  %barrier_resource, %barrier_timepoint = stream.timepoint.barrier %resource : !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK-NOT: stream.resource.dealloca
  util.return %barrier_timepoint : !stream.timepoint
}

// -----

// Tests that resources waited on do not cause deallocations as the timeline
// is disrupted.

// CHECK-LABEL: @ignoreAwaited
util.func private @ignoreAwaited(%input_timepoint: !stream.timepoint, %size: index) -> !stream.resource<transient> {
  // CHECK: %[[RESOURCE:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca
  %resource, %alloca_timepoint = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK-NOT: stream.resource.dealloca
  // CHECK: %[[READY_RESOURCE:.+]] = stream.timepoint.await %[[ALLOCA_TIMEPOINT]] => %[[RESOURCE]]
  %ready_resource = stream.timepoint.await %alloca_timepoint => %resource : !stream.resource<transient>{%size}
  // CHECK: util.return %[[READY_RESOURCE]]
  util.return %ready_resource : !stream.resource<transient>
}

// -----

// Tests that if stream.resource.retain/release ops are present the resource
// they are applied to is not touched but other resources do get handled.

// CHECK-LABEL: @ignoreRetainRelease
// CHECK-SAME: (%[[INPUT_TIMEPOINT:.+]]: !stream.timepoint, %[[SIZE:.+]]: index)
util.func private @ignoreRetainRelease(%input_timepoint: !stream.timepoint, %size: index) -> (!stream.timepoint, !stream.timepoint) {
  // CHECK: %[[IGNORED_RESOURCE:.+]], %[[IGNORED_TIMEPOINT:.+]] = stream.resource.alloca
  %ignored_resource, %ignored_timepoint = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: stream.resource.retain %[[IGNORED_RESOURCE]]
  stream.resource.retain %ignored_resource : !stream.resource<transient>{%size}
  // CHECK-NOT: stream.resource.dealloca
  // CHECK: %[[HANDLED_RESOURCE:.+]], %[[ALLOCA_TIMEPOINT:.+]] = stream.resource.alloca
  %handled_resource, %alloca_timepoint = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK: %[[DEALLOCA_TIMEPOINT:.+]] = stream.resource.dealloca origin await(%[[ALLOCA_TIMEPOINT]]) => %[[HANDLED_RESOURCE]] : !stream.resource<transient>{%[[SIZE]]} => !stream.timepoint
  // CHECK: util.return %[[IGNORED_TIMEPOINT]], %[[DEALLOCA_TIMEPOINT]]
  util.return %ignored_timepoint, %alloca_timepoint : !stream.timepoint, !stream.timepoint
}

// -----

// Tests that functions with more than one block are completely ignored.
// This is a limitation of the current pass implementation that operates on one
// block at a time. We could lighten this restriction by supporting certain
// cases of SSA dominance but a proper global analysis is the better way to
// handle things (and gives us calls/etc for free).

// CHECK-LABEL: @ignoreMultiBlockRegions
util.func private @ignoreMultiBlockRegions(%input_timepoint: !stream.timepoint, %size: index) -> !stream.timepoint {
  %resource, %alloca_timepoint = stream.resource.alloca uninitialized on(#hal.device.promise<@device>) await(%input_timepoint) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK-NOT: stream.resource.dealloca
  cf.br ^bb1
^bb1:
  util.return %alloca_timepoint : !stream.timepoint
}

// -----

// Tests that functions containing calls are completely ignored.
// This is a limitation of the current pass implementation (we don't have the
// global analysis to know whether the call mutates the resources or globals
// that the resources are aliased to).

// CHECK-LABEL: @ignoreRegionsWithCalls
util.func private @ignoreRegionsWithCalls(%input_timepoint: !stream.timepoint, %size: index) -> !stream.timepoint {
  %resource, %alloca_timepoint = stream.resource.alloca uninitialized on(#hal.device.promise<@device>) await(%input_timepoint) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK-NOT: stream.resource.dealloca
  util.call @some_func() : () -> ()
  util.return %alloca_timepoint : !stream.timepoint
}
util.func private @some_func() -> () {
  util.return
}

// -----

// TODO(benvanik): scf is something we should support even in this local pass
// in constrained scenarios (only for resources not used in regions, etc).
// For now using SCF will cause the entire parent block to be skipped.

// CHECK-LABEL: @ignoreSCF
util.func private @ignoreSCF(%input_timepoint: !stream.timepoint, %size: index) -> !stream.timepoint {
  %resource, %alloca_timepoint = stream.resource.alloca uninitialized on(#hal.device.promise<@device>) await(%input_timepoint) => !stream.resource<transient>{%size} => !stream.timepoint
  // CHECK-NOT: stream.resource.dealloca
  %cond = arith.constant 1 : i1
  scf.if %cond {
    scf.yield
  }
  util.return %alloca_timepoint : !stream.timepoint
}

// -----

// Tests that resources loaded from globals are treated as indeterminate.

util.global private @resource : !stream.resource<variable>
util.global private @timepoint : !stream.timepoint

// CHECK-LABEL: @ignoreGlobalLoad
util.func private @ignoreGlobalLoad(%size: index) -> !stream.timepoint {
  %resource = util.global.load @resource : !stream.resource<variable>
  %load_timepoint = util.global.load @timepoint : !stream.timepoint
  // CHECK: stream.cmd.execute
  %execute_timepoint = stream.cmd.execute await(%load_timepoint) => with(%resource as %capture : !stream.resource<variable>{%size}) {
  } => !stream.timepoint
  // CHECK-NOT: stream.resource.dealloca
  util.return %execute_timepoint : !stream.timepoint
}

// -----

// Tests that resources stored to globals are treated as indeterminate.

util.global private mutable @resource : !stream.resource<variable>
util.global private mutable @timepoint : !stream.timepoint

// CHECK-LABEL: @ignoreGlobalStore
util.func private @ignoreGlobalStore(%input_timepoint: !stream.timepoint, %size: index) -> () {
  %resource, %alloca_timepoint = stream.resource.alloca uninitialized await(%input_timepoint) => !stream.resource<variable>{%size} => !stream.timepoint
  // CHECK-NOT: stream.resource.dealloca
  util.global.store %resource, @resource : !stream.resource<variable>
  util.global.store %alloca_timepoint, @timepoint : !stream.timepoint
  util.return
}

// -----

// Tests that allocations explicitly marked as indeterminate don't get
// deallocated as if analysis decided so.

// CHECK-LABEL: @explicitIndeterminateAlloca
util.func private @explicitIndeterminateAlloca(%input_timepoint: !stream.timepoint, %size: index) -> () {
  %resource, %alloca_timepoint = stream.resource.alloca uninitialized indeterminate await(%input_timepoint) => !stream.resource<variable>{%size} => !stream.timepoint
  // CHECK-NOT: stream.resource.dealloca
  util.return
}

// -----

// Tests that resources loaded from parameters (or any other function that may
// allocate) are treated as indeterminate. This is not a strict requirement and
// we may want to support this in the future for parameter streaming.

// CHECK-LABEL: @ignoreNonAllocaProducers
util.func private @ignoreNonAllocaProducers(%size: index) -> () {
  %c0 = arith.constant 0 : i64
  // CHECK-NOT: stream.resource.dealloca
  %load_resource, %load_timepoint = stream.parameter.load {
    "scope"::"name"[%c0] : !stream.resource<constant>{%size}
  } => !stream.timepoint
  util.return
}
