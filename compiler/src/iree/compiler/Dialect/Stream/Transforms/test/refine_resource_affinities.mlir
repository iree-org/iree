// RUN: iree-opt --split-input-file --iree-stream-refine-resource-affinities %s | FileCheck %s


// CHECK-LABEL: @refine_alloca_dealloca
util.func public @refine_alloca_dealloca() {
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index

  // Allocation on device_a that will be used on device_b
  // CHECK: stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
  %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@device_a>) : !stream.resource<transient>{%c16} => !stream.timepoint

  %cmd_timepoint = stream.cmd.execute on(#hal.device.affinity<@device_b>) await(%result_timepoint) => with(%result as %arg0: !stream.resource<transient>{%c16}) {
    stream.cmd.dispatch @executable_b::@entry {
      rw %arg0[%c0 for %c16] : !stream.resource<transient>{%c16}
    }
  } => !stream.timepoint

  // CHECK: stream.resource.dealloca on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>) await(%{{.+}}) => %[[RESULT:.+]]
  %dealloca_timepoint = stream.resource.dealloca on(#hal.device.affinity<@device_a>) await(%cmd_timepoint) => %result : !stream.resource<transient>{%c16} => !stream.timepoint
  util.return
}

// -----

// CHECK-LABEL: @refine_alloca_multiple_devices
util.func public @refine_alloca_multiple_devices() {
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index
  // Allocation on device_a that will be used on device_b and device_c
  // CHECK: stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>, #hal.device.affinity<@device_c>]>)
  %shared_buffer, %shared_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@device_a>) : !stream.resource<transient>{%c16} => !stream.timepoint

  %cmd1_timepoint = stream.cmd.execute on(#hal.device.affinity<@device_b>) await(%shared_timepoint) => with(%shared_buffer as %arg0: !stream.resource<transient>{%c16}) {
    stream.cmd.dispatch @executable_b::@entry {
      rw %arg0[%c0 for %c16] : !stream.resource<transient>{%c16}
    }
  } => !stream.timepoint

  %cmd2_timepoint = stream.cmd.execute on(#hal.device.affinity<@device_c>) await(%cmd1_timepoint) => with(%shared_buffer as %arg0: !stream.resource<transient>{%c16}) {
    stream.cmd.dispatch @executable_c::@entry {
      ro %arg0[%c0 for %c16] : !stream.resource<transient>{%c16}
    }
  } => !stream.timepoint

  util.return
}

// -----

// CHECK-LABEL: @no_change_single_device_usage
util.func public @no_change_single_device_usage() {
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index

  // CHECK: stream.resource.alloca uninitialized on(#hal.device.affinity<@device_a>)
  %buffer, %buffer_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@device_a>) : !stream.resource<transient>{%c16} => !stream.timepoint

  %cmd_timepoint = stream.cmd.execute on(#hal.device.affinity<@device_a>) await(%buffer_timepoint) => with(%buffer as %arg0: !stream.resource<transient>{%c16}) {
    stream.cmd.dispatch @executable::@entry {
      rw %arg0[%c0 for %c16] : !stream.resource<transient>{%c16}
    }
  } => !stream.timepoint

  %dealloca_timepoint = stream.resource.dealloca on(#hal.device.affinity<@device_a>) await(%cmd_timepoint) => %buffer : !stream.resource<transient>{%c16} => !stream.timepoint
  util.return
}

// -----

// CHECK-LABEL: @refine_alloc_cross_device
util.func public @refine_alloc_cross_device() {
  %c16 = arith.constant 16 : index
  %c0 = arith.constant 0 : index

  // Allocation on device_a that will be used on device_b
  // CHECK: stream.resource.alloc uninitialized on(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
  %result = stream.resource.alloc uninitialized on(#hal.device.affinity<@device_a>) : !stream.resource<external>{%c16}

  %cmd_timepoint = stream.cmd.execute on(#hal.device.affinity<@device_b>) with(%result as %arg0: !stream.resource<external>{%c16}) {
    stream.cmd.dispatch @executable_b::@entry {
      rw %arg0[%c0 for %c16] : !stream.resource<external>{%c16}
    }
  } => !stream.timepoint

  util.return
}
