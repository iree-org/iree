// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(util.func(iree-stream-reify-reference-counting))' %s | FileCheck %s

// CHECK-LABEL: @asyncRetain
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index)
util.func private @asyncRetain(%resource: !stream.resource<*>, %size: index) -> !stream.resource<*> {
  // CHECK: stream.resource.retain %[[RESOURCE]] : !stream.resource<*>{%[[SIZE]]}
  %result = stream.async.retain %resource : !stream.resource<*>{%size}
  // CHECK: util.return %[[RESOURCE]]
  util.return %result : !stream.resource<*>
}

// -----

// CHECK-LABEL: @asyncReleaseImmediate
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index)
util.func private @asyncReleaseImmediate(%resource: !stream.resource<*>, %size: index) -> (!stream.resource<*>, !stream.timepoint) {
  // CHECK: %[[WAS_TERMINAL:.+]] = stream.resource.release %[[RESOURCE]] : !stream.resource<*>{%[[SIZE]]}
  // CHECK: %[[IF_TIMEPOINT:.+]] = scf.if %[[WAS_TERMINAL]]
  // CHECK-NEXT:   %[[DISCARD_TIMEPOINT:.+]] = stream.resource.dealloca origin %[[RESOURCE]] : !stream.resource<*>{%[[SIZE]]} => !stream.timepoint
  // CHECK-NEXT:   scf.yield %[[DISCARD_TIMEPOINT]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   %[[IMMEDIATE:.+]] = stream.timepoint.immediate
  // CHECK-NEXT:   scf.yield %[[IMMEDIATE]]
  %result, %result_timepoint = stream.async.release %resource : !stream.resource<*>{%size} => !stream.timepoint
  // CHECK: util.return %[[RESOURCE]], %[[IF_TIMEPOINT]]
  util.return %result, %result_timepoint : !stream.resource<*>, !stream.timepoint
}

// -----

// CHECK-LABEL: @asyncRelease
// CHECK-SAME: (%[[RESOURCE:.+]]: !stream.resource<*>, %[[SIZE:.+]]: index, %[[AWAIT_TIMEPOINT:.+]]: !stream.timepoint)
util.func private @asyncRelease(%resource: !stream.resource<*>, %size: index, %await_timepoint: !stream.timepoint) -> (!stream.resource<*>, !stream.timepoint) {
  // CHECK: %[[WAS_TERMINAL:.+]] = stream.resource.release %[[RESOURCE]] : !stream.resource<*>{%[[SIZE]]}
  // CHECK: %[[IF_TIMEPOINT:.+]] = scf.if %[[WAS_TERMINAL]]
  // CHECK-NEXT:   %[[DISCARD_TIMEPOINT:.+]] = stream.resource.dealloca origin await(%[[AWAIT_TIMEPOINT]]) => %[[RESOURCE]] : !stream.resource<*>{%[[SIZE]]} => !stream.timepoint
  // CHECK-NEXT:   scf.yield %[[DISCARD_TIMEPOINT]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   scf.yield %[[AWAIT_TIMEPOINT]]
  %result, %result_timepoint = stream.async.release await(%await_timepoint) => %resource : !stream.resource<*>{%size} => !stream.timepoint
  // CHECK: util.return %[[RESOURCE]], %[[IF_TIMEPOINT]]
  util.return %result, %result_timepoint : !stream.resource<*>, !stream.timepoint
}
