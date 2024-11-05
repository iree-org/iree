// RUN: iree-opt --split-input-file --allow-unregistered-dialect --iree-hal-conversion --cse --iree-hal-indirect-command-buffers=true %s | FileCheck %s

// Today all memory control operations are ignored and we're just left with
// the normal sequential execution barriers.

util.global private @device : !hal.device

// CHECK-LABEL: @cmdMemoryControl
util.func public @cmdMemoryControl(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  %0 = stream.cmd.execute on(#hal.device.affinity<@device>) with(%arg0 as %arg2: !stream.resource<transient>{%arg1}) {
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

// Tests that an execution region with a fill and any other op is converted to
// a command buffer instead of a queue operation.

util.global private @device : !hal.device

// CHECK-LABEL: @cmdFill
// CHECK-SAME: (%[[TARGET:.+]]: !hal.buffer, %[[TARGET_SIZE:.+]]: index)
util.func public @cmdFill(%target: !stream.resource<transient>, %target_size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c255_i32 = arith.constant 255 : i32
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  %signal = stream.cmd.execute once on(#hal.device.affinity<@device>) with(%target as %target_capture: !stream.resource<transient>{%target_size}) {
    // CHECK-NEXT: hal.command_buffer.fill_buffer<%[[CMD]] : !hal.command_buffer>
    // CHECK-SAME: target(%[[TARGET]] : !hal.buffer)[%c0, %c128]
    // CHECK-SAME: pattern(%c255_i32 : i32)
    // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
    stream.cmd.fill %c255_i32, %target_capture[%c0 for %c128] : i32 -> !stream.resource<transient>{%target_size}
    // CHECK-NEXT: hal.command_buffer.fill_buffer
    // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
    stream.cmd.fill %c255_i32, %target_capture[%c0 for %target_size] : i32 -> !stream.resource<transient>{%target_size}
  } => !stream.timepoint
  // CHECK-NEXT: hal.command_buffer.finalize<%[[CMD]]
  util.return %signal : !stream.timepoint
}

// -----

// Tests that an execution region with a single fill is converted to a queue
// operation instead of a command buffer. The extra flush is ignored as queue
// operations have implicit flushes (today).

util.global private @device : !hal.device

// CHECK-LABEL: @cmdFillOnQueue
// CHECK-SAME: (%[[TARGET:.+]]: !hal.buffer, %[[TARGET_SIZE:.+]]: index, %[[WAIT:.+]]: !hal.fence)
util.func public @cmdFillOnQueue(%target: !stream.resource<transient>, %target_size: index, %wait: !stream.timepoint) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c255_i32 = arith.constant 255 : i32
  // CHECK: %[[SIGNAL:.+]] = hal.fence.create
  // CHECK-NOT: hal.command_buffer.create
  %signal = stream.cmd.execute once on(#hal.device.affinity<@device>) await(%wait) => with(%target as %target_capture: !stream.resource<transient>{%target_size}) {
    // CHECK: hal.device.queue.fill<%{{.+}} : !hal.device>
    // CHECK-SAME: wait(%[[WAIT]]) signal(%[[SIGNAL]])
    // CHECK-SAME: target(%[[TARGET]] : !hal.buffer)[%c0] length(%c128)
    // CHECK-SAME: pattern(%c255_i32 : i32)
    stream.cmd.fill %c255_i32, %target_capture[%c0 for %c128] : i32 -> !stream.resource<transient>{%target_size}
    stream.cmd.flush %target_capture[%c0 for %c128] : !stream.resource<transient>{%target_size}
  } => !stream.timepoint
  // CHECK: util.return %[[SIGNAL]]
  util.return %signal : !stream.timepoint
}

// -----

// Tests that an execution region with a copy and any other op is converted to
// a command buffer instead of a queue operation.

util.global private @device : !hal.device

// CHECK-LABEL: @cmdCopy
// CHECK-SAME: (%[[SOURCE:.+]]: !hal.buffer, %[[SOURCE_SIZE:.+]]: index, %[[TARGET:.+]]: !hal.buffer, %[[TARGET_SIZE:.+]]: index)
util.func public @cmdCopy(%source: !stream.resource<transient>, %source_size: index, %target: !stream.resource<staging>, %target_size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  %signal = stream.cmd.execute once on(#hal.device.affinity<@device>) with(%source as %source_capture: !stream.resource<transient>{%source_size}, %target as %target_capture: !stream.resource<staging>{%target_size}) {
    // CHECK-NEXT: hal.command_buffer.copy_buffer<%[[CMD]] : !hal.command_buffer>
    // CHECK-SAME: source(%[[SOURCE]] : !hal.buffer)[%c0]
    // CHECK-SAME: target(%[[TARGET]] : !hal.buffer)[%c0]
    // CHECK-SAME: length(%c128)
    // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
    stream.cmd.copy %source_capture[%c0], %target_capture[%c0], %c128 : !stream.resource<transient>{%source_size} -> !stream.resource<staging>{%target_size}
    // CHECK-NEXT: hal.command_buffer.copy_buffer
    // CHECK-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
    stream.cmd.copy %source_capture[%c0], %target_capture[%c0], %target_size : !stream.resource<transient>{%source_size} -> !stream.resource<staging>{%target_size}
  } => !stream.timepoint
  // CHECK-NEXT: hal.command_buffer.finalize<%[[CMD]]
  util.return %signal : !stream.timepoint
}

// -----

// Tests that an execution region with a single copy is converted to a queue
// operation instead of a command buffer. The extra flush is ignored as queue
// operations have implicit flushes (today).

util.global private @device : !hal.device

// CHECK-LABEL: @cmdCopyOnQueue
// CHECK-SAME: (%[[SOURCE:.+]]: !hal.buffer, %[[SOURCE_SIZE:.+]]: index, %[[TARGET:.+]]: !hal.buffer, %[[TARGET_SIZE:.+]]: index, %[[WAIT:.+]]: !hal.fence)
util.func public @cmdCopyOnQueue(%source: !stream.resource<transient>, %source_size: index, %target: !stream.resource<staging>, %target_size: index, %wait: !stream.timepoint) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK-NOT: hal.command_buffer.create
  // CHECK: %[[SIGNAL:.+]] = hal.fence.create
  %signal = stream.cmd.execute once on(#hal.device.affinity<@device>) await(%wait) => with(%source as %source_capture: !stream.resource<transient>{%source_size}, %target as %target_capture: !stream.resource<staging>{%target_size}) {
    // CHECK: hal.device.queue.copy<%{{.+}} : !hal.device>
    // CHECK-SAME: wait(%[[WAIT]]) signal(%[[SIGNAL]])
    // CHECK-SAME: source(%[[SOURCE]] : !hal.buffer)[%c0]
    // CHECK-SAME: target(%[[TARGET]] : !hal.buffer)[%c0]
    // CHECK-SAME: length(%c128)
    stream.cmd.copy %source_capture[%c0], %target_capture[%c0], %c128 : !stream.resource<transient>{%source_size} -> !stream.resource<staging>{%target_size}
    stream.cmd.flush %target_capture[%c0 for %c128] : !stream.resource<staging>{%target_size}
  } => !stream.timepoint
  // CHECK: util.return %[[SIGNAL]]
  util.return %signal : !stream.timepoint
}

// -----

util.global private @device : !hal.device

// CHECK-LABEL: @cmdCollective
util.func public @cmdCollective(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<transient>, %arg3: index, %arg4: !stream.channel) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  %0 = stream.cmd.execute once on(#hal.device.affinity<@device>) with(%arg0 as %arg5: !stream.resource<transient>{%arg1}, %arg2 as %arg6: !stream.resource<transient>{%arg3}) {

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

util.global private @device : !hal.device

// CHECK-LABEL: @cmdExecute
util.func public @cmdExecute(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<staging>, %arg3: index, %arg4: !stream.timepoint) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  %0 = stream.cmd.execute once on(#hal.device.affinity<@device>) await(%arg4) => with(%arg0 as %arg5: !stream.resource<transient>{%arg1}, %arg2 as %arg6: !stream.resource<staging>{%arg3}) {
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
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer, Indirect>
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

util.global private @device : !hal.device
util.global private @constant_resource : !stream.resource<constant>
util.global private @constant_size : index

// CHECK-LABEL: @cmdDispatch
//  CHECK-SAME: (%[[ARG_RESOURCE:.+]]: !hal.buffer)
util.func public @cmdDispatch(%arg_resource: !stream.resource<external>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4_i32 = arith.constant 4 : i32
  %c5_i32 = arith.constant 5 : i32
  %c128 = arith.constant 128 : index
  // CHECK-DAG: %[[CONSTANT_RESOURCE:.+]] = util.global.load immutable @constant_resource
  %constant_resource = util.global.load immutable @constant_resource : !stream.resource<constant>
  %constant_size = util.global.load immutable @constant_size : index
  // CHECK: %[[ARG_SIZE:.+]] = arith.constant 200
  %arg_size = arith.constant 200 : index
  // CHECK-DAG: %[[DEVICE:.+]] = util.global.load immutable @device
  // CHECK: %[[MEMOIZED_CMD:.+]] = hal.device.memoize
  // CHECK: %[[CMD:.+]] = hal.command_buffer.create
  %0 = stream.cmd.execute on(#hal.device.affinity<@device>) with(%constant_resource as %constant_capture: !stream.resource<constant>{%constant_size}, %arg_resource as %arg_capture: !stream.resource<external>{%arg_size}) {
    // Switch for each executable variant by checking conditions and ranking:
    // CHECK: %[[CMD_DEVICE:.+]] = hal.command_buffer.device<%[[CMD]] : !hal.command_buffer>
    //  CHECK-DAG: %{{.+}}, %[[AARCH64_FORMAT:.+]] = hal.device.query<%[[CMD_DEVICE]] : !hal.device> key("hal.executable.format" :: "embedded-elf-aarch64")
    //  CHECK-DAG: %[[AARCH64_FEATURE:.+]] = scf.execute_region -> i1 {
    // CHECK-NEXT:   %{{.+}}, %[[FEATURE:.+]] = hal.device.query<%[[CMD_DEVICE]] : !hal.device> key("some" :: "feature")
    // CHECK-NEXT:   scf.yield %[[FEATURE]]
    // CHECK-NEXT: }
    //  CHECK-DAG: %[[AARCH64_SELECTED:.+]] = arith.andi %[[AARCH64_FORMAT]], %[[AARCH64_FEATURE]]
    //  CHECK-DAG: %{{.+}}, %[[X86_64_SELECTED:.+]] = hal.device.query<%[[CMD_DEVICE]] : !hal.device> key("hal.executable.format" :: "embedded-elf-x86_64")
    // CHECK: %[[VARIANT1:.+]] = arith.select %[[X86_64_SELECTED]], %c1
    // CHECK: %[[VARIANT0:.+]] = arith.select %[[AARCH64_SELECTED]], %c0, %[[VARIANT1]]
    // CHECK: scf.index_switch %[[VARIANT0]]
    // CHECK-NEXT: case 0 {

    // Inlined workgroup count calculation:
    // CHECK: %[[X:.+]] = affine.apply #map()[%c1]

    // Target executable/export:
    //  CHECK-DAG: %[[EXECUTABLE_0:.+]] = hal.executable.lookup
    // CHECK-SAME:     device(%[[CMD_DEVICE]] : !hal.device)
    // CHECK-SAME:     executable(@ex) : !hal.executable
    //  CHECK-DAG: %[[ORDINAL_0:.+]] = hal.executable.export.ordinal
    // CHECK-SAME:     target(@ex::@aarch64::@dispatch) : index

    // Dispatch:
    // CHECK: hal.command_buffer.dispatch<%[[CMD]]
    // CHECK-SAME: target(%[[EXECUTABLE_0]] : !hal.executable)[%[[ORDINAL_0]]]
    // CHECK-SAME: workgroups([%[[X]], %c1, %c1])
    // CHECK-SAME: constants([%c4_i32, %c5_i32])
    // CHECK-SAME: bindings([
    // CHECK-NEXT:   (%[[CONSTANT_RESOURCE]] : !hal.buffer)[%c0, %c128],
    // CHECK-NEXT:   (%c0 : index)[%c0, %c128]

    // Other variant, when selected:
    // CHECK: case 1 {
    // CHECK-DAG: %[[ORDINAL_1:.+]] = hal.executable.export.ordinal target(@ex::@x86_64::@dispatch)
    // CHECK: hal.command_buffer.dispatch<%[[CMD]]
    // CHECK-SAME: target({{.+}})[%[[ORDINAL_1]]]
    stream.cmd.dispatch {@ex::@aarch64::@dispatch, @ex::@x86_64::@dispatch}[%c1, %c2, %c3](%c4_i32, %c5_i32 : i32, i32) {
      ro %constant_capture[%c0 for %c128] : !stream.resource<constant>{%constant_size},
      wo %arg_capture[%c0 for %c128] : !stream.resource<external>{%arg_size}
    }
    // CHECK: hal.command_buffer.execution_barrier<%[[CMD]]
  } => !stream.timepoint
  // CHECK-NEXT: hal.command_buffer.finalize<%[[CMD]]
  //      CHECK: hal.device.queue.execute.indirect<%[[DEVICE]] : !hal.device> {{.+}} commands(%[[MEMOIZED_CMD]]) bindings([
  // CHECK-NEXT:   (%[[ARG_RESOURCE]] : !hal.buffer)[%c0, %[[ARG_SIZE]]]
  // CHECK-NEXT: ])
  util.return %0 : !stream.timepoint
}

// -----

// Tests conversion of streamable calls and function declarations.
// Expect a command buffer and a (binding table ordinal, buffer) + offset +
// length for each resource. Here we have one constant global that gets baked
// into the memoized command buffer and the two arguments are treated as
// indirect bindings.

util.global private @device : !hal.device

util.global private @global : !stream.resource<constant>

// CHECK: util.func private @cmdFunc(
// CHECK-SAME: %arg0: !hal.command_buffer,
stream.cmd.func private @cmdFunc(
    // CHECK-SAME: %arg1: index, %arg2: !hal.buffer, %arg3: index, %arg4: index,
    %arg0[%arg1 for %arg2]: !stream.resource<*>,
    // CHECK-SAME: %arg5: i32,
    %arg3: i32,
    // CHECK-SAME: %arg6: index, %arg7: !hal.buffer, %arg8: index, %arg9: index,
    %arg4[%arg5 for %arg6]: !stream.resource<*>,
    // CHECK-SAME: %arg10: !custom.type,
    %arg7: !custom.type,
    // CHECK-SAME: %arg11: index, %arg12: !hal.buffer, %arg13: index, %arg14: index
    %arg8[%arg9 for %arg10]: !stream.resource<*>)

// CHECK-LABEL: @cmdCall
util.func public @cmdCall(
    // CHECK-SAME: (%[[ARG_I32:.+]]: i32, %[[ARG_CUSTOM:.+]]: !custom.type,
    %arg_i32: i32, %arg_custom: !custom.type,
    // CHECK-SAME:  %[[ARG_RESOURCE0:.+]]: !hal.buffer, %[[ARG_RESOURCE1:.+]]: !hal.buffer)
    %arg_resource0: !stream.resource<transient>, %arg_resource1: !stream.resource<external>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[GLOBAL_RESOURCE:.+]] = util.global.load immutable @global
  %global_resource = util.global.load immutable @global : !stream.resource<constant>
  // CHECK-DAG: %[[GLOBAL_SIZE:.+]] = arith.constant 100
  %global_size = arith.constant 100 : index
  // CHECK-DAG: %[[ARG_SIZE0:.+]] = arith.constant 101
  %arg_size0 = arith.constant 101 : index
  // CHECK-DAG: %[[ARG_SIZE1:.+]] = arith.constant 102
  %arg_size1 = arith.constant 102 : index
  // CHECK: hal.device.memoize
  // CHECK: %[[COMMAND_BUFFER:.+]] = hal.command_buffer.create
  %timepoint = stream.cmd.execute on(#hal.device.affinity<@device>)
      with(%global_resource as %stream0: !stream.resource<constant>{%global_size}, %arg_resource0 as %stream1: !stream.resource<transient>{%arg_size0}, %arg_resource1 as %stream2: !stream.resource<external>{%arg_size1}) {
    // CHECK-DAG: %[[NULL_BUFFER:.+]] = util.null : !hal.buffer
    // CHECK: util.call @cmdFunc(%[[COMMAND_BUFFER]],
    stream.cmd.call @cmdFunc(
        // CHECK-SAME: %c0, %[[GLOBAL_RESOURCE]], %c0, %[[GLOBAL_SIZE]], %[[ARG_I32]],
        ro %stream0[%c0 for %global_size], %arg_i32,
        // CHECK-SAME: %c0, %[[NULL_BUFFER]], %c0, %[[ARG_SIZE0]], %[[ARG_CUSTOM]],
        rw %stream1[%c0 for %arg_size0], %arg_custom,
        // CHECK-SAME: %c1, %[[NULL_BUFFER]], %c0, %[[ARG_SIZE1]]
        wo %stream2[%c0 for %arg_size1]) :
        // CHECK-SAME: (!hal.command_buffer, index, !hal.buffer, index, index, i32, index, !hal.buffer, index, index, !custom.type, index, !hal.buffer, index, index) -> ()
        (!stream.resource<constant>{%global_size}, i32, !stream.resource<transient>{%arg_size0}, !custom.type, !stream.resource<external>{%arg_size1}) -> ()
  } => !stream.timepoint
  // CHECK: hal.device.queue.execute.indirect
  // CHECK-SAME: bindings([
  // CHECK-NEXT:   (%[[ARG_RESOURCE0]] : !hal.buffer)[%c0, %[[ARG_SIZE0]]],
  // CHECK-NEXT:   (%[[ARG_RESOURCE1]] : !hal.buffer)[%c0, %[[ARG_SIZE1]]]
  // CHECK-NEXT: ])
  util.return %timepoint : !stream.timepoint
}

// -----

// Tests that an operation specified to run on multiple queues ends up with the
// appropriate queue affinity mask. The final affinity is the result of ORing
// the target affinities (0b01 | 0b10 = 0b11 = 3).

util.global private @device : !hal.device

// CHECK-LABEL: @cmdExecuteAffinities
util.func public @cmdExecuteAffinities(%arg0: !stream.resource<transient>, %arg1: index, %arg2: !stream.resource<staging>, %arg3: index, %arg4: !stream.timepoint) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  // CHECK: hal.device.queue.copy
  // CHECK-SAME: affinity(%c3_i64)
  %0 = stream.cmd.execute once on(#hal.device.affinity<@device, [0, 1]>) await(%arg4) => with(%arg0 as %arg5: !stream.resource<transient>{%arg1}, %arg2 as %arg6: !stream.resource<staging>{%arg3}) {
    stream.cmd.copy %arg5[%c0], %arg6[%c0], %c128 : !stream.resource<transient>{%arg1} -> !stream.resource<staging>{%arg3}
  } => !stream.timepoint
  util.return %0 : !stream.timepoint
}
