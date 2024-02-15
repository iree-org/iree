// RUN: iree-opt --split-input-file --allow-unregistered-dialect --iree-hal-conversion %s | FileCheck %s

// Today all memory control operations are ignored and we're just left with
// the normal sequential execution barriers.

// CHECK-LABEL: @cmdMemoryControl
util.func public @cmdMemoryControl(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  %0 = stream.cmd.execute with(%arg0 as %arg2: !stream.resource<transient>{%arg1}) {
    // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
    stream.cmd.flush %arg2[%c0 for %c128] : !stream.resource<transient>{%arg1}
    // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
    stream.cmd.invalidate %arg2[%c0 for %c128] : !stream.resource<transient>{%arg1}
    // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
    stream.cmd.discard %arg2[%c0 for %c128] : !stream.resource<transient>{%arg1}
  } => !stream.timepoint
  // CHECK-NEXT: hal.command_buffer.finalize<%[[CMD]]
  util.return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @cmdFill
util.func public @cmdFill(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c255_i32 = arith.constant 255 : i32
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  %0 = stream.cmd.execute with(%arg0 as %arg2: !stream.resource<transient>{%arg1}) {
    // CHECK-NEXT: hal.command_buffer.fill_buffer<%[[CMD]] : !hal.command_buffer>
    // CHECK-SAME: target(%arg0 : !hal.buffer)[%c0, %c128]
    // CHECK-SAME: pattern(%c255_i32 : i32)
    stream.cmd.fill %c255_i32, %arg2[%c0 for %c128] : i32 -> !stream.resource<transient>{%arg1}
    // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
  } => !stream.timepoint
  // CHECK-NEXT: hal.command_buffer.finalize<%[[CMD]]
  util.return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @cmdCopy
util.func public @cmdCopy(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<staging>, %arg3: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  %0 = stream.cmd.execute with(%arg0 as %arg4: !stream.resource<transient>{%arg1}, %arg2 as %arg5: !stream.resource<staging>{%arg3}) {
    // CHECK-NEXT: hal.command_buffer.copy_buffer<%[[CMD]] : !hal.command_buffer>
    // CHECK-SAME: source(%arg0 : !hal.buffer)[%c0]
    // CHECK-SAME: target(%arg2 : !hal.buffer)[%c0]
    // CHECK-SAME: length(%c128)
    stream.cmd.copy %arg4[%c0], %arg5[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
    // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
  } => !stream.timepoint
  // CHECK-NEXT: hal.command_buffer.finalize<%[[CMD]]
  util.return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @cmdCollective
util.func public @cmdCollective(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<transient>, %arg3: index, %arg4: !stream.channel) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  %0 = stream.cmd.execute with(%arg0 as %arg5: !stream.resource<transient>{%arg1}, %arg2 as %arg6: !stream.resource<transient>{%arg3}) {

    // Out-of-place all-reduce:
    // CHECK-NEXT: hal.command_buffer.collective
    // CHECK-SAME: channel(%arg4 : !hal.channel)
    // CHECK-SAME: op(<all_reduce with sum : si8>)
    // CHECK-SAME: send(%arg0 : !hal.buffer)[%c0, %c128]
    // CHECK-SAME: recv(%arg2 : !hal.buffer)[%c0, %c128]
    // CHECK-SAME: count(%c128)
    stream.cmd.collective<all_reduce with sum : si8>[%c128] channel(%arg4) {
      ro %arg5[%c0 for %c128] : !stream.resource<transient>{%arg1},
      wo %arg6[%c0 for %c128] : !stream.resource<transient>{%arg3}
    }
    // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]

    // In-place all-reduce:
    // CHECK-NEXT: hal.command_buffer.collective
    // CHECK-SAME: channel(%arg4 : !hal.channel)
    // CHECK-SAME: op(<all_reduce with average : si8>)
    // CHECK-SAME: send(%arg0 : !hal.buffer)[%c0, %c128]
    // CHECK-SAME: recv(%arg0 : !hal.buffer)[%c0, %c128]
    // CHECK-SAME: count(%c128)
    stream.cmd.collective<all_reduce with average : si8>[%c128] channel(%arg4) {
      ro %arg5[%c0 for %c128] : !stream.resource<transient>{%arg1},
      wo %arg5[%c0 for %c128] : !stream.resource<transient>{%arg3}
    }
    // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]

    // Send:
    // CHECK-NEXT: hal.command_buffer.collective
    // CHECK-SAME: channel(%arg4 : !hal.channel)
    // CHECK-SAME: op(<send : si8>)
    // CHECK-SAME: send(%arg0 : !hal.buffer)[%c0, %c128]
    // CHECK-SAME: count(%c128)
    stream.cmd.collective<send : si8>[%c128] channel(%arg4) {
      ro %arg5[%c0 for %c128] : !stream.resource<transient>{%arg1}
    }
    // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]

    // Recv:
    // CHECK-NEXT: hal.command_buffer.collective
    // CHECK-SAME: channel(%arg4 : !hal.channel)
    // CHECK-SAME: op(<recv : si8>)
    // CHECK-SAME: recv(%arg0 : !hal.buffer)[%c0, %c128]
    // CHECK-SAME: count(%c128)
    stream.cmd.collective<recv : si8>[%c128] channel(%arg4) {
      wo %arg5[%c0 for %c128] : !stream.resource<transient>{%arg1}
    }
    // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]

  } => !stream.timepoint
  // CHECK-NEXT: hal.command_buffer.finalize<%[[CMD]]
  util.return %0 : !stream.timepoint
}

// -----

// NOTE: we don't currently do anything smart with the DAG because we aren't
// actually partitioning for that yet. This causes us to insert more barriers
// than we actually need and guard a lot more work than we otherwise would need
// to.

// CHECK-LABEL: @cmdExecute
util.func public @cmdExecute(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<staging>, %arg3: index, %arg4: !stream.timepoint) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  %0 = stream.cmd.execute await(%arg4) => with(%arg0 as %arg5: !stream.resource<transient>{%arg1}, %arg2 as %arg6: !stream.resource<staging>{%arg3}) {
    stream.cmd.concurrent {
      // CHECK-NEXT: hal.command_buffer.copy_buffer<%[[CMD]]
      stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
      // CHECK-NEXT: hal.command_buffer.copy_buffer<%[[CMD]]
      stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
      stream.cmd.serial {
        // CHECK-NEXT: hal.command_buffer.copy_buffer<%[[CMD]]
        // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
        stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
        // CHECK-NEXT: hal.command_buffer.copy_buffer<%[[CMD]]
        // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
        stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
      }
      // CHECK-NEXT: hal.command_buffer.copy_buffer<%[[CMD]]
      stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
      // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
    }
  } => !stream.timepoint
  // CHECK-NEXT: hal.command_buffer.finalize<%[[CMD]]
  // CHECK: %[[SIGNAL_FENCE:.+]] = hal.fence.create
  // CHECK: hal.device.queue.execute
  // CHECK-SAME: affinity(%c-1
  // CHECK-SAME: wait(%arg4)
  // CHECK-SAME: signal(%[[SIGNAL_FENCE]])
  // CHECK-SAME: commands([%[[CMD]]])
  // CHECK: util.return %[[SIGNAL_FENCE]]
  util.return %0 : !stream.timepoint
}

// -----

#executable_target_aarch64 = #hal.executable.target<"llvm-cpu", "embedded-elf-aarch64">
#executable_target_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
#device_target_cpu = #hal.device.target<"llvm-cpu", {
  executable_targets = [#executable_target_aarch64, #executable_target_x86_64]
}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<4, storage_buffer>
  ]>,
  #hal.descriptor_set.layout<1, bindings = [
    #hal.descriptor_set.binding<5, storage_buffer>
  ]>
]>
hal.executable private @ex {
  hal.executable.variant public @aarch64 target(#executable_target_aarch64) {
    hal.executable.condition(%device: !hal.device) -> i1 {
      %ok, %selected = hal.device.query<%device : !hal.device> key("some" :: "feature") : i1, i1
      hal.return %selected : i1
    }
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<CPUDefault>
    } {
    ^bb0(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index):  // no predecessors
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg0]
      hal.return %0, %c1, %c1 : index, index, index
    }
    builtin.module {
      // Opaque at this point (in some target-specific dialects).
    }
  }
  hal.executable.variant public @x86_64 target(#executable_target_x86_64) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<CPUDefault>
    } {
    ^bb0(%device: !hal.device, %arg0: index, %arg1: index, %arg2: index):  // no predecessors
      %c1 = arith.constant 1 : index
      %0 = affine.apply affine_map<()[s0] -> (s0 ceildiv 4)>()[%arg0]
      hal.return %0, %c1, %c1 : index, index, index
    }
    builtin.module {
      // Opaque at this point (in some target-specific dialects).
    }
  }
}

// CHECK-LABEL: @cmdDispatch
util.func public @cmdDispatch(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<external>, %arg3: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4_i32 = arith.constant 4 : i32
  %c5_i32 = arith.constant 5 : i32
  %c128 = arith.constant 128 : index
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  %0 = stream.cmd.execute with(%arg0 as %arg4: !stream.resource<transient>{%arg1}, %arg2 as %arg5: !stream.resource<external>{%arg3}) {
    // Switch for each executable variant by checking conditions and ranking:
    // CHECK: %[[DEVICE:.+]] = hal.command_buffer.device<%[[CMD]] : !hal.command_buffer>
    //  CHECK-DAG: %{{.+}}, %[[AARCH64_FORMAT:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.executable.format" :: "embedded-elf-aarch64")
    //  CHECK-DAG: %[[AARCH64_FEATURE:.+]] = scf.execute_region -> i1 {
    // CHECK-NEXT:   %{{.+}}, %[[FEATURE:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("some" :: "feature")
    // CHECK-NEXT:   scf.yield %[[FEATURE]]
    // CHECK-NEXT: }
    //  CHECK-DAG: %[[AARCH64_SELECTED:.+]] = arith.andi %[[AARCH64_FORMAT]], %[[AARCH64_FEATURE]]
    //  CHECK-DAG: %{{.+}}, %[[X86_64_SELECTED:.+]] = hal.device.query<%[[DEVICE]] : !hal.device> key("hal.executable.format" :: "embedded-elf-x86_64")
    // CHECK: %[[VARIANT1:.+]] = arith.select %[[X86_64_SELECTED]], %c1
    // CHECK: %[[VARIANT0:.+]] = arith.select %[[AARCH64_SELECTED]], %c0{{.+}}, %[[VARIANT1]]
    // CHECK: scf.index_switch %[[VARIANT0]]
    // CHECK-NEXT: case 0 {

    // Cache queries:
    //  CHECK-DAG:   %[[LAYOUT:.+]] = hal.pipeline_layout.lookup {{.+}} layout(#pipeline_layout)

    // Push constants:
    //  CHECK-DAG:   hal.command_buffer.push_constants<%[[CMD]]
    // CHECK-SAME:       layout(%[[LAYOUT]] : !hal.pipeline_layout)
    // CHECK-SAME:       offset(0)
    // CHECK-SAME:       values([%c4_i32, %c5_i32]) : i32, i32

    // Descriptor sets:
    //  CHECK-DAG:   hal.command_buffer.push_descriptor_set<%[[CMD]]
    // CHECK-SAME:       layout(%[[LAYOUT]] : !hal.pipeline_layout)[%c0
    // CHECK-NEXT:     %c4 = (%arg0 : !hal.buffer)[%c0, %c128]
    //  CHECK-DAG:   hal.command_buffer.push_descriptor_set<%[[CMD]]
    // CHECK-SAME:       layout(%[[LAYOUT]] : !hal.pipeline_layout)[%c1
    // CHECK-NEXT:     %c5 = (%arg2 : !hal.buffer)[%c0, %c128]

    // Inlined workgroup count calculation:
    // CHECK: %[[YZ:.+]] = arith.constant 1 : index
    // CHECK-NEXT: %[[X:.+]] = affine.apply #map()[%c1]

    // Dispatch:
    // CHECK: hal.command_buffer.dispatch.symbol<%[[CMD]]
    // CHECK-SAME: target(@ex::@aarch64::@dispatch)
    // CHECK-SAME: workgroups([%[[X]], %[[YZ]], %[[YZ]]])

    // Other variant, when selected:
    // CHECK: case 1 {
    // CHECK: hal.command_buffer.dispatch.symbol<%[[CMD]]
    // CHECK-SAME: target(@ex::@x86_64::@dispatch)
    stream.cmd.dispatch {@ex::@aarch64::@dispatch, @ex::@x86_64::@dispatch}[%c1, %c2, %c3](%c4_i32, %c5_i32 : i32, i32) {
      ro %arg4[%c0 for %c128] : !stream.resource<transient>{%arg1},
      wo %arg5[%c0 for %c128] : !stream.resource<external>{%arg3}
    } attributes {
      hal.interface.bindings = [
        #hal.interface.binding<0, 4>,
        #hal.interface.binding<1, 5>
      ]
    }
    // CHECK: hal.command_buffer.execution_barrier<%[[CMD]]
  } => !stream.timepoint
  // CHECK-NEXT: hal.command_buffer.finalize<%[[CMD]]
  util.return %0 : !stream.timepoint
}

// -----

// Tests conversion of streamable calls and function declarations.
// Expect a command buffer and a buffer + offset + length for each resource.

// CHECK: util.func private @cmdFunc(%arg0: !hal.command_buffer, %arg1: !hal.buffer, %arg2: index, %arg3: index, %arg4: i32, %arg5: !hal.buffer, %arg6: index, %arg7: index, %arg8: !custom.type, %arg9: !hal.buffer, %arg10: index, %arg11: index)
stream.cmd.func private @cmdFunc(%arg0[%arg1 for %arg2]: !stream.resource<*>, %arg3: i32, %arg4[%arg5 for %arg6]: !stream.resource<*>, %arg7: !custom.type, %arg8[%arg9 for %arg10]: !stream.resource<*>)

// CHECK-LABEL: @cmdCall
util.func public @cmdCall(%arg0: !stream.resource<external>, %arg1: i32, %arg2: !stream.resource<external>, %arg3: !custom.type, %arg4: !stream.resource<external>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[SIZE0:.+]] = arith.constant 100
  %size0 = arith.constant 100 : index
  // CHECK-DAG: %[[SIZE1:.+]] = arith.constant 101
  %size1 = arith.constant 101 : index
  // CHECK-DAG: %[[SIZE2:.+]] = arith.constant 102
  %size2 = arith.constant 102 : index
  // CHECK: %[[COMMAND_BUFFER:.+]] = hal.command_buffer.create
  %timepoint = stream.cmd.execute with(%arg0 as %stream0: !stream.resource<external>{%size0}, %arg2 as %stream1: !stream.resource<external>{%size1}, %arg4 as %stream2: !stream.resource<external>{%size2}) {
    // CHECK: util.call @cmdFunc(%[[COMMAND_BUFFER]], %arg0, %c0, %[[SIZE0]], %arg1, %arg2, %c0, %[[SIZE1]], %arg3, %arg4, %c0, %[[SIZE2]]) :
    // CHECK-SAME: (!hal.command_buffer, !hal.buffer, index, index, i32, !hal.buffer, index, index, !custom.type, !hal.buffer, index, index) -> ()
    stream.cmd.call @cmdFunc(ro %stream0[%c0 for %size0], %arg1, rw %stream1[%c0 for %size1], %arg3, wo %stream2[%c0 for %size2]) : (!stream.resource<external>{%size0}, i32, !stream.resource<external>{%size1}, !custom.type, !stream.resource<external>{%size2}) -> ()
  } => !stream.timepoint
  util.return %timepoint : !stream.timepoint
}

// -----

// Tests that an operation specified to run on multiple queues ends up with the
// appropriate queue affinity mask. The final affinity is the result of ORing
// the target affinities (0b01 | 0b10 = 0b11 = 3).

// CHECK-LABEL: @cmdExecuteAffinities
util.func public @cmdExecuteAffinities(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<staging>, %arg3: index, %arg4: !stream.timepoint) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  %0 = stream.cmd.execute on(#hal.affinity.queue<[0, 1]>) await(%arg4) => with(%arg0 as %arg5: !stream.resource<transient>{%arg1}, %arg2 as %arg6: !stream.resource<staging>{%arg3}) {
    stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
  } => !stream.timepoint
  // CHECK: hal.device.queue.execute
  // CHECK-SAME: affinity(%c3_i64)
  // CHECK-SAME: commands([%[[CMD]]])
  util.return %0 : !stream.timepoint
}
