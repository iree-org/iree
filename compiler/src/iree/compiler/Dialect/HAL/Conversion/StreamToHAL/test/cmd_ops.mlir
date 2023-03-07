// RUN: iree-opt --split-input-file --iree-hal-conversion %s | FileCheck %s

// Today all memory control operations are ignored and we're just left with
// the normal sequential execution barriers.

// CHECK-LABEL: @cmdMemoryControl
func.func @cmdMemoryControl(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint {
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
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @cmdFill
func.func @cmdFill(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint {
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
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @cmdCopy
func.func @cmdCopy(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<staging>, %arg3: index) -> !stream.timepoint {
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
  return %0 : !stream.timepoint
}

// -----

// CHECK-LABEL: @cmdCollective
func.func @cmdCollective(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<transient>, %arg3: index, %arg4: !stream.channel) -> !stream.timepoint {
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
  return %0 : !stream.timepoint
}

// -----

// NOTE: we don't currently do anything smart with the DAG because we aren't
// actually partitioning for that yet. This causes us to insert more barriers
// than we actually need and guard a lot more work than we otherwise would need
// to.

// CHECK-LABEL: @cmdExecute
func.func @cmdExecute(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<staging>, %arg3: index, %arg4: !stream.timepoint) -> !stream.timepoint {
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
  // CHECK: return %[[SIGNAL_FENCE]]
  return %0 : !stream.timepoint
}

// -----

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
#device_target_cpu = #hal.device.target<"llvm-cpu", {
  executable_targets = [#executable_target_embedded_elf_x86_64_]
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
  hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
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
func.func @cmdDispatch(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<external>, %arg3: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4_i32 = arith.constant 4 : i32
  %c5_i32 = arith.constant 5 : i32
  %c128 = arith.constant 128 : index
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  %0 = stream.cmd.execute with(%arg0 as %arg4: !stream.resource<transient>{%arg1}, %arg2 as %arg5: !stream.resource<external>{%arg3}) {
    // Switch for each executable variant:
    // CHECK: hal.device.switch
    // CHECK-NEXT: #hal.device.match.executable.format<"embedded-elf-x86_64">

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
    // CHECK-SAME: target(@ex::@embedded_elf_x86_64::@dispatch)
    // CHECK-SAME: workgroups([%[[X]], %[[YZ]], %[[YZ]]])
    stream.cmd.dispatch @ex::@embedded_elf_x86_64::@dispatch[%c1, %c2, %c3](%c4_i32, %c5_i32 : i32, i32) {
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
  return %0 : !stream.timepoint
}

// -----

// Tests that an operation specified to run on multiple queues ends up with the
// appropriate queue affinity mask. The final affinity is the result of ORing
// the target affinities (0b01 | 0b10 = 0b11 = 3).

// CHECK-LABEL: @cmdExecuteAffinities
func.func @cmdExecuteAffinities(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<staging>, %arg3: index, %arg4: !stream.timepoint) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  %0 = stream.cmd.execute on(#hal.affinity.queue<[0, 1]>) await(%arg4) => with(%arg0 as %arg5: !stream.resource<transient>{%arg1}, %arg2 as %arg6: !stream.resource<staging>{%arg3}) {
    stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
  } => !stream.timepoint
  // CHECK: hal.device.queue.execute
  // CHECK-SAME: affinity(%c3_i64)
  // CHECK-SAME: commands([%[[CMD]]])
  return %0 : !stream.timepoint
}
