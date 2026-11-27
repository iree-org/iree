// RUN: iree-opt --split-input-file --verify-each --pass-pipeline='builtin.module(util.func(iree-stream-materialize-copy-on-write),iree-stream-elide-async-copies,iree-stream-schedule-allocation)' %s | FileCheck %s

module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {unified_memory = true})
  ]>
} {
  // CHECK-LABEL: @local_handoff
  // CHECK: %[[STORAGE:[^ ,]+]], %[[READY:[^ ]+]] = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
  // CHECK-NOT: stream.resource.alloca
  // CHECK: %[[DONE:[^ ]+]] = stream.cmd.execute on(#hal.device.affinity<@device_a>) await(%[[READY]]) => with(%[[STORAGE]] as %[[CAPTURE:[^ :]+]]: !stream.resource<transient>
  // CHECK: stream.cmd.fill %{{.+}}, %[[CAPTURE]]
  // CHECK-NOT: stream.cmd.copy
  // CHECK: stream.cmd.flush to(#hal.device.affinity<@device_b>) %[[CAPTURE]]
  // CHECK: util.optimization_barrier %[[STORAGE]]
  // CHECK: util.optimization_barrier %[[DONE]]
  util.func public @local_handoff(%size: index, %value: i32) {
    %result, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with() -> !stream.resource<transient>{%size} {
      %source = stream.async.splat %value : i32 -> !stream.resource<transient>{%size}
      %transfer = stream.async.transfer %source : !stream.resource<transient>{%size} -> to(#hal.device.affinity<@device_b>) !stream.resource<transient>{%size} storage = reuse
      stream.yield %transfer : !stream.resource<transient>{%size}
    } => !stream.timepoint
    util.optimization_barrier %result : !stream.resource<transient>
    util.optimization_barrier %done : !stream.timepoint
    util.return
  }
}

// -----

// The first region owns its new storage. The second region is its only
// consumer, so mutating the reused result needs no protective allocation.
// CHECK-LABEL: @owned_reuse_then_mutate
// CHECK: %[[STORAGE:[^ ,]+]], %[[READY:[^ ]+]] = stream.resource.alloca uninitialized on(#hal.device.affinity<@device_a>)
// CHECK: %[[PRODUCED:[^ ]+]] = stream.cmd.execute on(#hal.device.affinity<@device_a>) await(%[[READY]]) => with(%[[STORAGE]] as %[[FIRST_CAPTURE:[^ :]+]]: !stream.resource<transient>
// CHECK: stream.cmd.fill %{{.+}}, %[[FIRST_CAPTURE]]
// CHECK-NOT: stream.resource.alloca
// CHECK: %[[MUTATED:[^ ]+]] = stream.cmd.execute on(#hal.device.affinity<@device_a>) await(%[[PRODUCED]]) => with(%[[STORAGE]] as %[[SECOND_CAPTURE:[^ :]+]]: !stream.resource<transient>
// CHECK-NOT: stream.cmd.copy
// CHECK: stream.cmd.fill %{{.+}}, %[[SECOND_CAPTURE]]
// CHECK: util.optimization_barrier %[[STORAGE]]
// CHECK: util.optimization_barrier %[[MUTATED]]
util.func public @owned_reuse_then_mutate(%size: index, %initial: i32, %new_value: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %source, %produced = stream.async.execute on(#hal.device.affinity<@device_a>) with() -> !stream.resource<transient>{%size} {
    %fresh = stream.async.splat %initial : i32 -> !stream.resource<transient>{%size}
    stream.yield %fresh : !stream.resource<transient>{%size}
  } => !stream.timepoint
  %result, %mutated = stream.async.execute on(#hal.device.affinity<@device_a>) await(%produced) => with(%source as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
    %transfer = stream.async.transfer %capture : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size} storage = reuse
    %updated = stream.async.fill %new_value, %transfer[%c0 to %size for %c1] : i32 -> %transfer as !stream.resource<transient>{%size}
    stream.yield %updated : !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.optimization_barrier %result : !stream.resource<transient>
  util.optimization_barrier %mutated : !stream.timepoint
  util.return
}

// -----

// The one copy here protects the original source from a later write through
// the reused transfer result; the transfer itself remains zero copy.
// CHECK-LABEL: @reuse_then_mutate(
// CHECK-SAME: %[[SOURCE:[^:]+]]: !stream.resource<transient>
// CHECK: %[[CLONE_STORAGE:[^ ,]+]], %[[CLONE_READY:[^ ]+]] = stream.resource.alloca uninitialized
// CHECK-NOT: stream.resource.alloca
// CHECK: %[[DONE:[^ ]+]] = stream.cmd.execute on(#hal.device.affinity<@device_a>) await(%[[CLONE_READY]]) => with(%[[SOURCE]] as %[[SOURCE_CAPTURE:[^ :]+]]: !stream.resource<transient>{{[^,]*}}, %[[CLONE_STORAGE]] as %[[CLONE_CAPTURE:[^ :]+]]: !stream.resource<transient>
// CHECK: stream.cmd.copy %[[SOURCE_CAPTURE]]{{.*}}, %[[CLONE_CAPTURE]]
// CHECK-NOT: stream.cmd.copy
// CHECK: stream.cmd.fill %{{.+}}, %[[CLONE_CAPTURE]]
// CHECK-NOT: stream.cmd.copy
// CHECK: util.optimization_barrier %[[SOURCE]], %[[CLONE_STORAGE]]
// CHECK: util.optimization_barrier %[[DONE]]
util.func public @reuse_then_mutate(%source: !stream.resource<transient>, %size: index, %value: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %results:2, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with(%source as %capture: !stream.resource<transient>{%size}) -> (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) {
    %transfer = stream.async.transfer %capture : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size} storage = reuse
    %updated = stream.async.fill %value, %transfer[%c0 to %size for %c1] : i32 -> %transfer as !stream.resource<transient>{%size}
    stream.yield %capture, %updated : !stream.resource<transient>{%size}, !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.optimization_barrier %results#0, %results#1 : !stream.resource<transient>, !stream.resource<transient>
  util.optimization_barrier %done : !stream.timepoint
  util.return
}

// -----

// A write through the original source also needs its own copy while the
// transfer result continues to refer to the original allocation.
// CHECK-LABEL: @reuse_then_mutate_source(
// CHECK-SAME: %[[SOURCE:[^:]+]]: !stream.resource<transient>
// CHECK: %[[CLONE_STORAGE:[^ ,]+]], %[[CLONE_READY:[^ ]+]] = stream.resource.alloca uninitialized
// CHECK-NOT: stream.resource.alloca
// CHECK: %[[DONE:[^ ]+]] = stream.cmd.execute on(#hal.device.affinity<@device_a>) await(%[[CLONE_READY]]) => with(%[[SOURCE]] as %[[SOURCE_CAPTURE:[^ :]+]]: !stream.resource<transient>{{[^,]*}}, %[[CLONE_STORAGE]] as %[[CLONE_CAPTURE:[^ :]+]]: !stream.resource<transient>
// CHECK: stream.cmd.copy %[[SOURCE_CAPTURE]]{{.*}}, %[[CLONE_CAPTURE]]
// CHECK-NOT: stream.cmd.copy
// CHECK: stream.cmd.fill %{{.+}}, %[[CLONE_CAPTURE]]
// CHECK-NOT: stream.cmd.copy
// CHECK: util.optimization_barrier %[[SOURCE]], %[[CLONE_STORAGE]]
// CHECK: util.optimization_barrier %[[DONE]]
util.func public @reuse_then_mutate_source(%source: !stream.resource<transient>, %size: index, %value: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %results:2, %done = stream.async.execute on(#hal.device.affinity<@device_a>) with(%source as %capture: !stream.resource<transient>{%size}) -> (!stream.resource<transient>{%size}, !stream.resource<transient>{%size}) {
    %transfer = stream.async.transfer %capture : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size} storage = reuse
    %updated = stream.async.fill %value, %capture[%c0 to %size for %c1] : i32 -> %capture as !stream.resource<transient>{%size}
    stream.yield %transfer, %updated : !stream.resource<transient>{%size}, !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.optimization_barrier %results#0, %results#1 : !stream.resource<transient>, !stream.resource<transient>
  util.optimization_barrier %done : !stream.timepoint
  util.return
}

// -----

// With no shared-memory topology, an explicit copy keeps its target buffer.
// CHECK-LABEL: @discrete_copy
// CHECK: %[[SOURCE:[^ ,]+]], %[[SOURCE_READY:[^ ]+]] = stream.resource.alloca uninitialized
// CHECK: %[[TARGET:[^ ,]+]], %[[TARGET_READY:[^ ]+]] = stream.resource.alloca uninitialized
// CHECK: stream.cmd.execute
// CHECK: stream.cmd.copy
// CHECK: stream.cmd.flush to(#hal.device.affinity<@device_b>)
// CHECK: util.optimization_barrier %[[TARGET]]
util.func public @discrete_copy(%size: index) {
  %source, %ready = stream.resource.alloca uninitialized on(#hal.device.affinity<@device_a>) : !stream.resource<transient>{%size} => !stream.timepoint
  %result, %done = stream.async.execute on(#hal.device.affinity<@device_a>) await(%ready) => with(%source as %capture: !stream.resource<transient>{%size}) -> !stream.resource<transient>{%size} {
    %transfer = stream.async.transfer %capture : !stream.resource<transient>{%size} -> to(#hal.device.affinity<@device_b>) !stream.resource<transient>{%size} storage = copy
    stream.yield %transfer : !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.optimization_barrier %result : !stream.resource<transient>
  util.optimization_barrier %done : !stream.timepoint
  util.return
}
