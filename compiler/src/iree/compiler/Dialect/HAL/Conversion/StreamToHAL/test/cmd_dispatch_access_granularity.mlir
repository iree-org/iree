// RUN: iree-opt --split-input-file --iree-hal-conversion --canonicalize --iree-hal-indirect-command-buffers=false %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-hal-conversion --canonicalize --iree-hal-indirect-command-buffers=true %s | FileCheck %s --check-prefix=INDIRECT

// A device that accesses storage in 4-byte units gets bindings that declare
// the padding reserved for them, up to the end of their buffer.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, Indirect>,
  #hal.pipeline.binding<storage_buffer, Indirect>
]>
hal.executable private @ex {
  hal.executable.variant public @variant target(#hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
    }
  }
}

util.global private @device = #hal.device.target<"word", {stream.min_access_granularity = 4 : i64}> : !hal.device

// CHECK-LABEL: @wordGranular
//  CHECK-SAME: (%[[SLAB:.+]]: !hal.buffer, %[[OUTPUT:.+]]: !hal.buffer)
// INDIRECT-LABEL: @wordGranular
//  INDIRECT-SAME: (%[[SLAB:.+]]: !hal.buffer, %[[OUTPUT:.+]]: !hal.buffer)
util.func public @wordGranular(%slab: !stream.resource<transient>, %output: !stream.resource<external>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c64 = arith.constant 64 : index
  %c192 = arith.constant 192 : index
  // An external buffer may be longer than its tensor, up to its own length.
  // CHECK: %[[CAPACITY:.+]] = hal.buffer.length<%[[OUTPUT]] : !hal.buffer> : index
  // CHECK: %[[OWNED:.+]] = arith.minui %[[CAPACITY]], %c8
  // INDIRECT: %[[CAPACITY:.+]] = hal.buffer.length<%[[OUTPUT]] : !hal.buffer> : index
  // INDIRECT: %[[SPAN:.+]] = arith.minui %[[CAPACITY]], %c8
  %0 = stream.cmd.execute on(#hal.device.affinity<@device>) with(%slab as %slab_capture: !stream.resource<transient>{%c192}, %output as %output_capture: !stream.resource<external>{%c6}) {
    // CHECK: hal.command_buffer.dispatch
    // CHECK-SAME: bindings([
    // CHECK-NEXT:   (%[[SLAB]] : !hal.buffer)[%c64, %c8],
    // CHECK-NEXT:   (%[[OUTPUT]] : !hal.buffer)[%c0, %[[OWNED]]]
    // The command reaching the external end takes the slot bound per call.
    // INDIRECT: hal.command_buffer.dispatch
    // INDIRECT-SAME: bindings([
    // INDIRECT-NEXT:   (%c0 : index)[%c64, %c8],
    // INDIRECT-NEXT:   (%c1 : index)[%c0, %c-1]
    // INDIRECT: hal.device.queue.execute.indirect
    // INDIRECT-SAME: bindings([
    // INDIRECT-NEXT:   (%[[SLAB]] : !hal.buffer)[%c0, %c192],
    // INDIRECT-NEXT:   (%[[OUTPUT]] : !hal.buffer)[%c0, %[[SPAN]]]
    stream.cmd.dispatch @ex::@variant::@dispatch {
      ro %slab_capture[%c64 for %c6] : !stream.resource<transient>{%c192},
      wo %output_capture[%c0 for %c6] : !stream.resource<external>{%c6}
    }
  } => !stream.timepoint
  util.return %0 : !stream.timepoint
}

// -----

// An empty binding stays empty; a dynamic one rounds within its resource.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @ex {
  hal.executable.variant public @variant target(#hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
    }
  }
}

util.global private @device = #hal.device.target<"word", {stream.min_access_granularity = 4 : i64}> : !hal.device

// CHECK-LABEL: @emptyAndDynamic
//  CHECK-SAME: (%{{.+}}: !hal.buffer, %[[OFFSET:.+]]: index, %[[LENGTH:.+]]: index, %[[SIZE:.+]]: index)
util.func public @emptyAndDynamic(%slab: !stream.resource<transient>, %offset: index, %length: index, %size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c192 = arith.constant 192 : index
  %0 = stream.cmd.execute on(#hal.device.affinity<@device>) with(%slab as %slab_capture: !stream.resource<transient>{%size}) {
    // CHECK: %[[END:.+]] = arith.addi %[[OFFSET]], %[[LENGTH]]
    // CHECK: %[[ALIGNED:.+]] = util.align %[[END]], %c4
    // CHECK: %[[OWNED:.+]] = arith.minui %[[ALIGNED]], %[[SIZE]]
    // CHECK: %[[OWNED_LENGTH:.+]] = arith.subi %[[OWNED]], %[[OFFSET]]
    // CHECK: %[[EMPTY:.+]] = arith.cmpi eq, %[[LENGTH]], %c0
    // CHECK: %[[LENGTH_OR_OWNED:.+]] = arith.select %[[EMPTY]], %c0, %[[OWNED_LENGTH]]
    // CHECK: hal.command_buffer.dispatch
    // CHECK-SAME: bindings([
    // CHECK-NEXT:   (%{{.+}} : !hal.buffer)[%c6, %c0],
    // CHECK-NEXT:   (%{{.+}} : !hal.buffer)[%[[OFFSET]], %[[LENGTH_OR_OWNED]]]
    stream.cmd.dispatch @ex::@variant::@dispatch {
      ro %slab_capture[%c6 for %c0] : !stream.resource<transient>{%size},
      wo %slab_capture[%offset for %length] : !stream.resource<transient>{%size}
    }
  } => !stream.timepoint
  util.return %0 : !stream.timepoint
}

// -----

// A byte-granular device keeps the logical lengths.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @ex {
  hal.executable.variant public @variant target(#hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
    }
  }
}

util.global private @device = #hal.device.target<"byte"> : !hal.device

// CHECK-LABEL: @byteGranular
util.func public @byteGranular(%slab: !stream.resource<transient>) -> !stream.timepoint {
  %c6 = arith.constant 6 : index
  %c64 = arith.constant 64 : index
  %c192 = arith.constant 192 : index
  %0 = stream.cmd.execute on(#hal.device.affinity<@device>) with(%slab as %slab_capture: !stream.resource<transient>{%c192}) {
    // CHECK: hal.command_buffer.dispatch
    // CHECK-SAME: bindings([
    // CHECK-NEXT:   (%{{.+}} : !hal.buffer)[%c64, %c6]
    stream.cmd.dispatch @ex::@variant::@dispatch {
      rw %slab_capture[%c64 for %c6] : !stream.resource<transient>{%c192}
    }
  } => !stream.timepoint
  util.return %0 : !stream.timepoint
}

// -----

// An external slice of a larger allocation is bounded by its own view.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @ex {
  hal.executable.variant public @variant target(#hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb">) {
    hal.executable.export public @dispatch ordinal(0) layout(#pipeline_layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
    }
  }
}

util.global private @device = #hal.device.target<"word", {stream.min_access_granularity = 4 : i64}> : !hal.device

// CHECK-LABEL: @externalSlice
//  CHECK-SAME: (%[[ALLOCATION:.+]]: !hal.buffer)
util.func public @externalSlice(%allocation: !stream.resource<external>) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c6 = arith.constant 6 : index
  %c64 = arith.constant 64 : index
  %c192 = arith.constant 192 : index
  %slice = stream.resource.subview %allocation[%c64] : !stream.resource<external>{%c192} -> !stream.resource<external>{%c6}
  %0 = stream.cmd.execute on(#hal.device.affinity<@device>) with(%slice as %slice_capture: !stream.resource<external>{%c6}) {
    // CHECK: %[[SLICE:.+]] = hal.buffer.subspan<%[[ALLOCATION]] : !hal.buffer>[%c64, %c6]
    // CHECK-NOT: hal.buffer.length<%[[ALLOCATION]]
    // CHECK: %[[CAPACITY:.+]] = hal.buffer.length<%[[SLICE]] : !hal.buffer> : index
    // CHECK: %[[OWNED:.+]] = arith.minui %[[CAPACITY]], %c8
    // CHECK: hal.command_buffer.dispatch
    // CHECK-SAME: bindings([
    // CHECK-NEXT:   (%[[ALLOCATION]] : !hal.buffer)[%c64, %[[OWNED]]]
    stream.cmd.dispatch @ex::@variant::@dispatch {
      wo %slice_capture[%c0 for %c6] : !stream.resource<external>{%c6}
    }
  } => !stream.timepoint
  util.return %0 : !stream.timepoint
}
