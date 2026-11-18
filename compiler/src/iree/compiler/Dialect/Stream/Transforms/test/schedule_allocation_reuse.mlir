// RUN: iree-opt --split-input-file --verify-each --iree-stream-schedule-allocation %s | FileCheck %s

module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {unified_memory = true})
  ]>
} {
  // CHECK-LABEL: @reuse_outgoing
  // CHECK: %[[SOURCE:.+]], %[[READY:.+]] = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
  // CHECK-NOT: stream.resource.alloca
  // CHECK: %[[DONE:.+]] = stream.cmd.execute on(#hal.device.affinity<@device_a>) await(%[[READY]]) => with(%[[SOURCE]] as %[[CAPTURE:.+]]: !stream.resource<transient>
  // CHECK-NOT: stream.cmd.copy
  // CHECK: stream.cmd.flush to(#hal.device.affinity<@device_b>) %[[CAPTURE]]
  // CHECK: util.optimization_barrier %[[SOURCE]]
  // CHECK: util.optimization_barrier %[[DONE]]
  util.func public @reuse_outgoing(%size: index) {
    %source, %ready = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>) : !stream.resource<transient>{%size} => !stream.timepoint
    %result, %done = stream.async.execute on(#hal.device.affinity<@device_a>) await(%ready) => with(%source as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
      %transfer = stream.async.transfer %capture : !stream.resource<transient>{%size} -> to(#hal.device.affinity<@device_b>) !stream.resource<transient>{%size} storage = reuse
      stream.yield %transfer : !stream.resource<transient>{%size}
    } => !stream.timepoint
    util.optimization_barrier %result : !stream.resource<transient>
    util.optimization_barrier %done : !stream.timepoint
    util.return
  }
}

// -----

module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {unified_memory = true})
  ]>
} {
  // CPU/device A produces into shared storage and flushes the handoff; device
  // B reads it only after A's completion, using the same backing allocation.
  // CHECK-LABEL: @two_region_device_handoff
  // CHECK: %[[STORAGE:[^ ,]+]], %[[READY:[^ ]+]] = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
  // CHECK-NOT: stream.resource.alloca
  // CHECK: %[[A_DONE:[^ ]+]] = stream.cmd.execute on(#hal.device.affinity<@device_a>) await(%[[READY]]) => with(%[[STORAGE]] as %[[A_CAPTURE:[^ :]+]]: !stream.resource<transient>
  // CHECK: stream.cmd.fill %{{.+}}, %[[A_CAPTURE]]
  // CHECK-NOT: stream.cmd.copy
  // CHECK: stream.cmd.flush to(#hal.device.affinity<@device_b>) %[[A_CAPTURE]]
  // CHECK: %[[B_DONE:[^ ]+]] = stream.cmd.execute on(#hal.device.affinity<@device_b>) await(%[[A_DONE]]) => with(%[[STORAGE]] as %[[B_CAPTURE:[^ :]+]]: !stream.resource<transient>
  // CHECK-NOT: stream.cmd.copy
  // CHECK: stream.cmd.dispatch
  // CHECK: ro %[[B_CAPTURE]][
  // CHECK-NOT: stream.cmd.copy
  // CHECK: %{{[^ ]+}} = util.optimization_barrier %[[STORAGE]]
  // CHECK: util.optimization_barrier %[[B_DONE]]
  util.func public @two_region_device_handoff(%size: index, %value: i32) {
    %c0 = arith.constant 0 : index
    %source, %a_done = stream.async.execute on(#hal.device.affinity<@device_a>) with() -> !stream.resource<transient>{%size} {
      %fresh = stream.async.splat %value : i32 -> !stream.resource<transient>{%size}
      %transfer = stream.async.transfer %fresh : !stream.resource<transient>{%size} -> to(#hal.device.affinity<@device_b>) !stream.resource<transient>{%size} storage = reuse
      stream.yield %transfer : !stream.resource<transient>{%size}
    } => !stream.timepoint
    %result, %b_done = stream.async.execute on(#hal.device.affinity<@device_b>) await(%a_done) => with(%source as %b_capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
      stream.async.dispatch @ex::@consume(%b_capture[%c0 to %size for %size]) : (!stream.resource<transient>{%size}) -> ()
      stream.yield %b_capture : !stream.resource<transient>{%size}
    } => !stream.timepoint
    util.optimization_barrier %result : !stream.resource<transient>
    util.optimization_barrier %b_done : !stream.timepoint
    util.return
  }
}

// -----

module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {transparent_access = true})
  ]>
} {
  // CHECK-LABEL: @reuse_incoming
  // CHECK: %[[SOURCE:.+]], %[[READY:.+]] = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
  // CHECK-NOT: stream.resource.alloca
  // CHECK: stream.cmd.execute on(#hal.device.affinity<@device_b>) await(%[[READY]]) => with(%[[SOURCE]] as %[[CAPTURE:.+]]: !stream.resource<transient>
  // CHECK: stream.cmd.invalidate from(#hal.device.affinity<@device_a>) %[[CAPTURE]]
  // CHECK-NOT: stream.cmd.copy
  // CHECK: util.optimization_barrier %[[SOURCE]]
  util.func public @reuse_incoming(%size: index) {
    %source, %ready = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>) : !stream.resource<transient>{%size} => !stream.timepoint
    %result, %done = stream.async.execute on(#hal.device.affinity<@device_b>) await(%ready) => with(%source as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
      %transfer = stream.async.transfer %capture : !stream.resource<transient>{%size} from(#hal.device.affinity<@device_a>) -> !stream.resource<transient>{%size} storage = reuse
      stream.yield %transfer : !stream.resource<transient>{%size}
    } => !stream.timepoint
    util.optimization_barrier %result : !stream.resource<transient>
    util.optimization_barrier %done : !stream.timepoint
    util.return
  }
}

// -----

// A local producer writes directly into the allocation returned to callers.
// CHECK-LABEL: @reuse_local_producer
// CHECK: %[[STORAGE:.+]], %[[READY:.+]] = stream.resource.alloca uninitialized
// CHECK-NOT: stream.resource.alloca
// CHECK: stream.cmd.execute on(#hal.device.affinity<@device_a>) await(%[[READY]]) => with(%[[STORAGE]] as %[[CAPTURE:.+]]: !stream.resource<transient>
// CHECK: stream.cmd.fill %{{.+}}, %[[CAPTURE]]
// CHECK-NOT: stream.cmd.copy
// CHECK: util.optimization_barrier %[[STORAGE]]
util.func public @reuse_local_producer(%size: index, %value: i32) {
  %result, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with() -> !stream.resource<transient>{%size} {
    %source = stream.async.splat %value : i32 -> !stream.resource<transient>{%size}
    %transfer = stream.async.transfer %source : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size} storage = reuse
    stream.yield %transfer : !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.optimization_barrier %result : !stream.resource<transient>
  util.optimization_barrier %done : !stream.timepoint
  util.return
}

// -----

module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {unified_memory = true})
  ]>
} {
  // A local producer and its transfer result use one allocation selected for
  // both devices. The producer fill, flush, and completion timepoint remain.
  // CHECK-LABEL: @reuse_local_cross_device
  // CHECK: %[[STORAGE:.+]], %[[READY:.+]] = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
  // CHECK-NOT: stream.resource.alloca
  // CHECK: %[[DONE:.+]] = stream.cmd.execute on(#hal.device.affinity<@device_a>) await(%[[READY]]) => with(%[[STORAGE]] as %[[CAPTURE:.+]]: !stream.resource<transient>
  // CHECK: stream.cmd.fill %{{.+}}, %[[CAPTURE]]
  // CHECK-NOT: stream.cmd.copy
  // CHECK: stream.cmd.flush to(#hal.device.affinity<@device_b>) %[[CAPTURE]]
  // CHECK: util.optimization_barrier %[[STORAGE]]
  // CHECK: util.optimization_barrier %[[DONE]]
  util.func public @reuse_local_cross_device(%size: index, %value: i32) {
    %result, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with() -> !stream.resource<transient>{%size} {
      %source = stream.async.splat %value : i32 -> !stream.resource<transient>{%size}
      %transfer = stream.async.transfer %source : !stream.resource<transient>{%size} -> to(#hal.device.affinity<@device_b>) !stream.resource<transient>{%size} storage = reuse
      stream.yield %transfer : !stream.resource<transient>{%size}
    } => !stream.timepoint
    util.optimization_barrier %result : !stream.resource<transient>
    util.optimization_barrier %done : !stream.timepoint
    util.return
  }

  // The source and handoff result both escape but still share one reservation.
  // CHECK-LABEL: @reuse_local_cross_device_two_results
  // CHECK: %[[STORAGE:.+]], %[[READY:.+]] = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
  // CHECK-NOT: stream.resource.alloca
  // CHECK: stream.cmd.execute on(#hal.device.affinity<@device_a>) await(%[[READY]])
  // CHECK: stream.cmd.fill
  // CHECK-NOT: stream.cmd.copy
  // CHECK: stream.cmd.flush to(#hal.device.affinity<@device_b>)
  // CHECK: util.optimization_barrier %[[STORAGE]], %[[STORAGE]]
  util.func public @reuse_local_cross_device_two_results(%size: index, %value: i32) {
    %results:2, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with() -> (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) {
      %source = stream.async.splat %value : i32 -> !stream.resource<transient>{%size}
      %transfer = stream.async.transfer %source : !stream.resource<transient>{%size} -> to(#hal.device.affinity<@device_b>) !stream.resource<transient>{%size} storage = reuse
      stream.yield %transfer, %source : !stream.resource<transient>{%size}, !stream.resource<transient>{%size}
    } => !stream.timepoint
    util.optimization_barrier %results#0, %results#1 : !stream.resource<transient>, !stream.resource<transient>
    util.optimization_barrier %done : !stream.timepoint
    util.return
  }

  // Source-first result ordering must select the same full shared reservation.
  // CHECK-LABEL: @reuse_local_cross_device_source_first
  // CHECK: %[[STORAGE:.+]], %[[READY:.+]] = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
  // CHECK-NOT: stream.resource.alloca
  // CHECK: stream.cmd.execute on(#hal.device.affinity<@device_a>) await(%[[READY]])
  // CHECK: stream.cmd.fill
  // CHECK-NOT: stream.cmd.copy
  // CHECK: stream.cmd.flush to(#hal.device.affinity<@device_b>)
  // CHECK: util.optimization_barrier %[[STORAGE]], %[[STORAGE]]
  util.func public @reuse_local_cross_device_source_first(%size: index, %value: i32) {
    %results:2, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with() -> (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) {
      %source = stream.async.splat %value : i32 -> !stream.resource<transient>{%size}
      %transfer = stream.async.transfer %source : !stream.resource<transient>{%size} -> to(#hal.device.affinity<@device_b>) !stream.resource<transient>{%size} storage = reuse
      stream.yield %source, %transfer : !stream.resource<transient>{%size}, !stream.resource<transient>{%size}
    } => !stream.timepoint
    util.optimization_barrier %results#0, %results#1 : !stream.resource<transient>, !stream.resource<transient>
    util.optimization_barrier %done : !stream.timepoint
    util.return
  }
}

// -----

// CHECK-LABEL: @reuse_subview
// CHECK-NOT: stream.resource.alloca
// CHECK: stream.resource.subview %[[SOURCE:.+]][%[[OFFSET:.+]]]
// CHECK: stream.cmd.execute on(#hal.device.affinity<@device_a>) with(%[[SOURCE]] as %[[CAPTURE:.+]]: !stream.resource<transient>
// CHECK-NOT: stream.cmd.copy
// CHECK: util.optimization_barrier
util.func public @reuse_subview(%source: !stream.resource<transient>, %size: index, %offset: index, %length: index) {
  %result, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with(%source as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%length} {
    %view = stream.resource.subview %capture[%offset] : !stream.resource<transient>{%size} -> !stream.resource<transient>{%length}
    %transfer = stream.async.transfer %view : !stream.resource<transient>{%length} -> !stream.resource<transient>{%length} storage = reuse
    stream.yield %transfer : !stream.resource<transient>{%length}
  } => !stream.timepoint
  util.optimization_barrier %result : !stream.resource<transient>
  util.optimization_barrier %done : !stream.timepoint
  util.return
}

// -----

// Both SSA results keep the same backing allocation and completion timepoint.
// CHECK-LABEL: @reuse_multiple_consumers
// CHECK-NOT: stream.resource.alloca
// CHECK: %[[DONE:.+]] = stream.cmd.execute
// CHECK-NOT: stream.cmd.copy
// CHECK: util.optimization_barrier %[[SOURCE:.+]], %[[SOURCE]]
// CHECK: util.optimization_barrier %[[DONE]]
util.func public @reuse_multiple_consumers(%source: !stream.resource<transient>, %size: index) {
  %results:2, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with(%source as %capture: !stream.resource<transient>{%size}) -> (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) {
    %transfer = stream.async.transfer %capture : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size} storage = reuse
    stream.yield %capture, %transfer : !stream.resource<transient>{%size}, !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.optimization_barrier %results#0, %results#1 : !stream.resource<transient>, !stream.resource<transient>
  util.optimization_barrier %done : !stream.timepoint
  util.return
}

// -----

// A local producer and its reuse result can both escape without reserving two
// buffers. This also exercises a result processed before its source result.
// CHECK-LABEL: @reuse_local_two_results
// CHECK: %[[STORAGE:.+]], %[[READY:.+]] = stream.resource.alloca uninitialized
// CHECK-NOT: stream.resource.alloca
// CHECK: stream.cmd.execute on(#hal.device.affinity<@device_a>) await(%[[READY]])
// CHECK-NOT: stream.cmd.copy
// CHECK: util.optimization_barrier %[[STORAGE]], %[[STORAGE]]
util.func public @reuse_local_two_results(%size: index, %value: i32) {
  %results:2, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with() -> (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) {
    %source = stream.async.splat %value : i32 -> !stream.resource<transient>{%size}
    %transfer = stream.async.transfer %source : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size} storage = reuse
    stream.yield %transfer, %source : !stream.resource<transient>{%size}, !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.optimization_barrier %results#0, %results#1 : !stream.resource<transient>, !stream.resource<transient>
  util.optimization_barrier %done : !stream.timepoint
  util.return
}

// -----

// A discrete transfer with an explicit copy policy retains its destination.
// CHECK-LABEL: @copy_discrete_devices
// CHECK: %[[SOURCE:.+]], %[[SOURCE_READY:.+]] = stream.resource.alloca uninitialized
// CHECK: %[[TARGET:.+]], %[[TARGET_READY:.+]] = stream.resource.alloca uninitialized
// CHECK: stream.cmd.execute
// CHECK: stream.cmd.copy
// CHECK: stream.cmd.flush to(#hal.device.affinity<@device_b>)
// CHECK: util.optimization_barrier %[[TARGET]]
util.func public @copy_discrete_devices(%size: index) {
  %source, %ready = stream.resource.alloca uninitialized on(#hal.device.affinity<@device_a>) : !stream.resource<transient>{%size} => !stream.timepoint
  %result, %done = stream.async.execute on(#hal.device.affinity<@device_a>) await(%ready) => with(%source as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
    %transfer = stream.async.transfer %capture : !stream.resource<transient>{%size} -> to(#hal.device.affinity<@device_b>) !stream.resource<transient>{%size} storage = copy
    stream.yield %transfer : !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.optimization_barrier %result : !stream.resource<transient>
  util.optimization_barrier %done : !stream.timepoint
  util.return
}

// -----

// A view yielded after the handoff keeps its offset in the source allocation.
// CHECK-LABEL: @reuse_result_subview
// CHECK-NOT: stream.resource.alloca
// CHECK: stream.resource.subview %[[SOURCE:.+]][%[[OFFSET:.+]]]
// CHECK: stream.cmd.execute on(#hal.device.affinity<@device_a>) with(%[[SOURCE]]
// CHECK-NOT: stream.cmd.copy
// CHECK: util.optimization_barrier
util.func public @reuse_result_subview(%source: !stream.resource<transient>, %size: index, %offset: index, %length: index) {
  %result, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with(%source as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%length} {
    %transfer = stream.async.transfer %capture : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size} storage = reuse
    %view = stream.resource.subview %transfer[%offset] : !stream.resource<transient>{%size} -> !stream.resource<transient>{%length}
    stream.yield %view : !stream.resource<transient>{%length}
  } => !stream.timepoint
  util.optimization_barrier %result : !stream.resource<transient>
  util.optimization_barrier %done : !stream.timepoint
  util.return
}
