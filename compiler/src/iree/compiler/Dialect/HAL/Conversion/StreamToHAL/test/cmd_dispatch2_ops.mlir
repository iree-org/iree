// RUN: iree-opt --split-input-file --iree-hal-conversion --cse --iree-hal-indirect-command-buffers=true --iree-hal-experimental-dispatch2=true %s | FileCheck %s

#executable_target_aarch64 = #hal.executable.target<"llvm-cpu", "embedded-elf-aarch64">
#executable_target_x86_64 = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer, Indirect>
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

util.global private @device : !hal.device
util.global private @constant_resource : !stream.resource<constant>
util.global private @constant_size : index

// CHECK-LABEL: @cmdDispatch
//  CHECK-SAME: (%[[ARG_RESOURCE:.+]]: !hal.buffer, %[[ARG_SIZE:.+]]: index)
util.func public @cmdDispatch(%arg_resource: !stream.resource<external>, %arg_size: index) -> !stream.timepoint {
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
    // CHECK: hal.command_buffer.dispatch2<%[[CMD]]
    // CHECK-SAME: target(%[[EXECUTABLE_0]] : !hal.executable)[%[[ORDINAL_0]]]
    // CHECK-SAME: workgroups([%[[X]], %c1, %c1])
    // CHECK-SAME: constants([%c4_i32, %c5_i32])
    // CHECK-SAME: bindings([
    // CHECK-NEXT:   (%[[CONSTANT_RESOURCE]] : !hal.buffer)[%c0, %c128],
    // CHECK-NEXT:   (%c0 : index)[%c0, %c128]

    // Other variant, when selected:
    // CHECK: case 1 {
    // CHECK-DAG: %[[ORDINAL_1:.+]] = hal.executable.export.ordinal target(@ex::@x86_64::@dispatch)
    // CHECK: hal.command_buffer.dispatch2<%[[CMD]]
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
