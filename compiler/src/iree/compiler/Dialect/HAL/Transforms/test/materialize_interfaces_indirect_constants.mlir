// RUN: iree-opt --split-input-file --iree-hal-materialize-interfaces --iree-hal-indirect-constant-bindings=false %s | FileCheck %s --check-prefixes=CHECK,DIRECT
// RUN: iree-opt --split-input-file --iree-hal-materialize-interfaces --iree-hal-indirect-constant-bindings=true %s | FileCheck %s --check-prefixes=CHECK,INDIRECT

// Tests that bindings of constants dispatched from reusable execution regions
// are marked Indirect on devices with `indirect_constant_bindings = true` and on
// all devices with --iree-hal-indirect-constant-bindings.

util.global private @device_indirect = #hal.device.target<"cpu", {indirect_constant_bindings = true}, [
  #hal.executable.target<"llvm-cpu", "x86_64">
]> : !hal.device
util.global private @device_direct = #hal.device.target<"cpu", [
  #hal.executable.target<"llvm-cpu", "x86_64">
]> : !hal.device

// DIRECT-DAG: #[[LAYOUT_DIRECT:[a-z_0-9]+]] = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>, #hal.pipeline.binding<storage_buffer, Indirect>]
// DIRECT-DAG: #[[LAYOUT_INDIRECT:[a-z_0-9]+]] = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>]
// INDIRECT: #[[LAYOUT_INDIRECT:[a-z_0-9]+]] = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>]

// CHECK: hal.executable private @ex_indirect
// CHECK: hal.executable.export public @entry ordinal(0) layout(#[[LAYOUT_INDIRECT]])
stream.executable private @ex_indirect {
  stream.executable.export public @entry workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
  builtin.module {
    func.func @entry(%arg0: !stream.binding, %arg1: !stream.binding) {
      return
    }
  }
}

// CHECK: hal.executable private @ex_direct
// DIRECT: hal.executable.export public @entry ordinal(0) layout(#[[LAYOUT_DIRECT]])
// INDIRECT: hal.executable.export public @entry ordinal(0) layout(#[[LAYOUT_INDIRECT]])
stream.executable private @ex_direct {
  stream.executable.export public @entry workgroups() -> (index, index, index) {
    %c1 = arith.constant 1 : index
    stream.return %c1, %c1, %c1 : index, index, index
  }
  builtin.module {
    func.func @entry(%arg0: !stream.binding, %arg1: !stream.binding) {
      return
    }
  }
}

util.global private @constant_resource : !stream.resource<constant>
util.global private @constant_size : index

util.func public @main_indirect(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint attributes {
  stream.affinity = #hal.device.affinity<@device_indirect>
} {
  %c0 = arith.constant 0 : index
  %constant_resource = util.global.load immutable @constant_resource : !stream.resource<constant>
  %constant_size = util.global.load immutable @constant_size : index
  %0 = stream.cmd.execute with(%constant_resource as %arg2: !stream.resource<constant>{%constant_size}, %arg0 as %arg3: !stream.resource<transient>{%arg1}) {
    stream.cmd.dispatch @ex_indirect::@entry {
      ro %arg2[%c0 for %constant_size] : !stream.resource<constant>{%constant_size},
      wo %arg3[%c0 for %arg1] : !stream.resource<transient>{%arg1}
    }
  } => !stream.timepoint
  util.return %0 : !stream.timepoint
}

util.func public @main_direct(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint attributes {
  stream.affinity = #hal.device.affinity<@device_direct>
} {
  %c0 = arith.constant 0 : index
  %constant_resource = util.global.load immutable @constant_resource : !stream.resource<constant>
  %constant_size = util.global.load immutable @constant_size : index
  %0 = stream.cmd.execute with(%constant_resource as %arg2: !stream.resource<constant>{%constant_size}, %arg0 as %arg3: !stream.resource<transient>{%arg1}) {
    stream.cmd.dispatch @ex_direct::@entry {
      ro %arg2[%c0 for %constant_size] : !stream.resource<constant>{%constant_size},
      wo %arg3[%c0 for %arg1] : !stream.resource<transient>{%arg1}
    }
  } => !stream.timepoint
  util.return %0 : !stream.timepoint
}
