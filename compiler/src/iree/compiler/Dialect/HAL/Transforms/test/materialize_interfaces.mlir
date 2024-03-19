// RUN: iree-opt --split-input-file --iree-hal-materialize-interfaces %s | FileCheck %s

// Tests an executable with a workgroup count region specified.

module attributes {hal.device.targets = [
  #hal.device.target<"llvm-cpu", [
    #hal.executable.target<"llvm-cpu", "arm_64">,
    #hal.executable.target<"llvm-cpu", "x86_64">
  ]>
]} {
  // CHECK: #pipeline_layout = #hal.pipeline.layout<
  // CHECK-SAME: push_constants = 1
  // CHECK-SAME: sets = [
  // CHECK-SAME:   <0, bindings = [
  // CHECK-SAME:     <0, storage_buffer, ReadOnly>
  // CHECK-SAME:     <1, storage_buffer, ReadOnly>
  // CHECK-SAME:     <2, storage_buffer>

  // CHECK: hal.executable private @ex
  // CHECK:   hal.executable.variant public @arm_64 target(#executable_target_arm_64
  // CHECK:     hal.executable.export public @entry ordinal(0) layout(#pipeline_layout)
  // CHECK-SAME:   hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]
  // CHECK-NEXT: ^bb0(%[[DEVICE:.+]]: !hal.device, %[[ARG0:.+]]: index, %[[ARG1:.+]]: index):
  // CHECK-NEXT:   hal.return %[[ARG0]], %[[ARG1]], %[[ARG0]] : index, index, index
  // CHECK-NEXT: }
  // CHECK:     builtin.module
  // CHECK-NEXT:  func.func private @extern_func()
  // CHECK-NEXT:  func.func @entry
  // CHECK:   hal.executable.variant public @x86_64 target(#executable_target_x86_64
  // CHECK:     hal.executable.export public @entry ordinal(0) layout(#pipeline_layout)
  // CHECK-SAME:   hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>]
  // CHECK-NEXT: ^bb0(%[[DEVICE:.+]]: !hal.device, %[[ARG0:.+]]: index, %[[ARG1:.+]]: index):
  // CHECK-NEXT:   hal.return %[[ARG0]], %[[ARG1]], %[[ARG0]] : index, index, index
  // CHECK-NEXT: }
  // CHECK:     builtin.module
  // CHECK-NEXT:  func.func private @extern_func()

  // CHECK-NEXT:  func.func @entry
  stream.executable private @ex {
    stream.executable.export public @entry workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      stream.return %arg0, %arg1, %arg0 : index, index, index
    }
    builtin.module {
      func.func private @extern_func()
      func.func @entry(%operand: i32, %arg0: !stream.binding {stream.alignment = 64 : index}, %arg1: !stream.binding {stream.alignment = 64 : index}, %arg2: !stream.binding {stream.alignment = 64 : index}) {
        return
      }
    }
  }
  util.func public @main(%arg0: !stream.resource<constant>, %arg1: !stream.resource<transient>, %arg2: index, %arg3: i32) -> !stream.resource<transient> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = stream.resource.alloc uninitialized : !stream.resource<transient>{%arg2}
    %1 = stream.cmd.execute with(%arg0 as %arg4: !stream.resource<constant>{%arg2}, %arg1 as %arg5: !stream.resource<transient>{%arg2}, %0 as %arg6: !stream.resource<transient>{%arg2}) {
      // CHECK: stream.cmd.dispatch
      // CHECK-SAME: @ex::@arm_64::@entry
      // CHECK-SAME: @ex::@x86_64::@entry
      stream.cmd.dispatch @ex::@entry[%c1, %c2](%arg3 : i32) {
        ro %arg4[%c0 for %arg2] : !stream.resource<constant>{%arg2},
        ro %arg5[%c0 for %arg2] : !stream.resource<transient>{%arg2},
        wo %arg6[%c0 for %arg2] : !stream.resource<transient>{%arg2}
      }
    } => !stream.timepoint
    %2 = stream.timepoint.await %1 => %0 : !stream.resource<transient>{%arg2}
    util.return %2 : !stream.resource<transient>
  }
}

// -----

// Tests that executable variants are expanded based on what devices they are
// dispatched on.

module attributes {
  // The default device when none is specified.
  // Functions and scopes can override the target device.
  hal.device.targets = [
    #hal.device.target<"cpu", [
      #hal.executable.target<"llvm-cpu", "arm_64">,
      #hal.executable.target<"llvm-cpu", "x86_64">
    ]>
  ]
} {
  // CHECK: hal.executable private @ex
  // CHECK:   hal.executable.variant public @arm_64
  // CHECK:   hal.executable.variant public @riscv_32
  // CHECK:   hal.executable.variant public @x86_64
  stream.executable private @ex {
    stream.executable.export public @entry workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      stream.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @entry(%arg0: !stream.binding {stream.alignment = 64 : index}) {
        return
      }
    }
  }
  // This function uses the default HAL device targeting arm_64 and x86_64.
  // CHECK-LABEL: @using_default
  util.func public @using_default(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint {
    %c0 = arith.constant 0 : index
    %0 = stream.cmd.execute with(%arg0 as %arg2: !stream.resource<transient>{%arg1}) {
      // CHECK: stream.cmd.dispatch
      // CHECK-SAME: @ex::@arm_64::@entry
      // CHECK-NOT: @ex::@riscv_32::@entry
      // CHECK-SAME: @ex::@x86_64::@entry
      stream.cmd.dispatch @ex::@entry {
        rw %arg2[%c0 for %arg1] : !stream.resource<transient>{%arg1}
      }
    } => !stream.timepoint
    util.return %0 : !stream.timepoint
  }
  // This function is specialized to only run on only riscv_32 and should
  // not get assigned the arm_64/x86_64 variant entry points.
  // CHECK-LABEL: @using_specialized
  util.func public @using_specialized(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint attributes {
    hal.device.targets = [
      #hal.device.target<"cpu", [
        #hal.executable.target<"llvm-cpu", "riscv_32">
      ]>
    ]
  } {
    %c0 = arith.constant 0 : index
    %0 = stream.cmd.execute with(%arg0 as %arg2: !stream.resource<transient>{%arg1}) {
      // CHECK: stream.cmd.dispatch
      // CHECK-NOT: @ex::@arm_64::@entry
      // CHECK-SAME: @ex::@riscv_32::@entry
      // CHECK-NOT: @ex::@x86_64::@entry
      stream.cmd.dispatch @ex::@entry {
        rw %arg2[%c0 for %arg1] : !stream.resource<transient>{%arg1}
      }
    } => !stream.timepoint
    util.return %0 : !stream.timepoint
  }
}

// -----

// Tests an already-specified executable source op is expanded into the variants
// specified by the target configuration. These source executables may come from
// hand-authored code or other dialects that perform interface assignment
// themselves.

module attributes {
  // The default device when none is specified.
  // Functions and scopes can override the target device.
  hal.device.targets = [
    #hal.device.target<"cpu", [
      #hal.executable.target<"llvm-cpu", "arm_64">,
      #hal.executable.target<"llvm-cpu", "x86_64">
    ]>
  ]
} {
  // CHECK: hal.executable private @ex
  // CHECK:   hal.executable.variant public @arm_64
  // CHECK:   hal.executable.variant public @riscv_32
  // CHECK:   hal.executable.variant public @x86_64
  hal.executable.source private @ex {
    hal.executable.export public @entry layout(#hal.pipeline.layout<push_constants = 0, sets = [
      #hal.descriptor_set.layout<0, bindings = [
        #hal.descriptor_set.binding<0, storage_buffer>
      ]>
    ]>)
    builtin.module {
      func.func @entry() {
        return
      }
    }
  }
  // This function uses the default HAL device targeting arm_64 and x86_64.
  // CHECK-LABEL: @using_default
  util.func public @using_default(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint {
    %c0 = arith.constant 0 : index
    %0 = stream.cmd.execute with(%arg0 as %arg2: !stream.resource<transient>{%arg1}) {
      // CHECK: stream.cmd.dispatch
      // CHECK-SAME: @ex::@arm_64::@entry
      // CHECK-NOT: @ex::@riscv_32::@entry
      // CHECK-SAME: @ex::@x86_64::@entry
      stream.cmd.dispatch @ex::@entry {
        rw %arg2[%c0 for %arg1] : !stream.resource<transient>{%arg1}
      }
    } => !stream.timepoint
    util.return %0 : !stream.timepoint
  }
  // This function is specialized to only run on only riscv_32 and should
  // not get assigned the arm_64/x86_64 variant entry points.
  // CHECK-LABEL: @using_specialized
  util.func public @using_specialized(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint attributes {
    hal.device.targets = [
      #hal.device.target<"cpu", [
        #hal.executable.target<"llvm-cpu", "riscv_32">
      ]>
    ]
  } {
    %c0 = arith.constant 0 : index
    %0 = stream.cmd.execute with(%arg0 as %arg2: !stream.resource<transient>{%arg1}) {
      // CHECK: stream.cmd.dispatch
      // CHECK-NOT: @ex::@arm_64::@entry
      // CHECK-SAME: @ex::@riscv_32::@entry
      // CHECK-NOT: @ex::@x86_64::@entry
      stream.cmd.dispatch @ex::@entry {
        rw %arg2[%c0 for %arg1] : !stream.resource<transient>{%arg1}
      }
    } => !stream.timepoint
    util.return %0 : !stream.timepoint
  }
}

// -----

// Tests that a hal.executable.source op gets expanded to all default targets
// when it's public in addition to any ones from dispatch sites.

module attributes {
  hal.device.targets = [
    #hal.device.target<"cpu", [
      #hal.executable.target<"llvm-cpu", "arm_64">,
      #hal.executable.target<"llvm-cpu", "x86_64">
    ]>
  ]
} {
  // CHECK: hal.executable public @ex
  // CHECK:   hal.executable.variant public @arm_64
  // CHECK:   hal.executable.variant public @riscv_32
  // CHECK:   hal.executable.variant public @x86_64
  hal.executable.source public @ex {
    hal.executable.export public @entry layout(#hal.pipeline.layout<push_constants = 0, sets = [
      #hal.descriptor_set.layout<0, bindings = [
        #hal.descriptor_set.binding<0, storage_buffer>
      ]>
    ]>)
    builtin.module {
      func.func @entry() {
        return
      }
    }
  }
  // CHECK-LABEL: @using_specialized
  util.func public @using_specialized(%arg0: !stream.resource<transient>, %arg1: index) -> !stream.timepoint attributes {
    hal.device.targets = [
      #hal.device.target<"cpu", [
        #hal.executable.target<"llvm-cpu", "riscv_32">
      ]>
    ]
  } {
    %c0 = arith.constant 0 : index
    %0 = stream.cmd.execute with(%arg0 as %arg2: !stream.resource<transient>{%arg1}) {
      // CHECK: stream.cmd.dispatch
      // CHECK-NOT: @ex::@arm_64::@entry
      // CHECK-SAME: @ex::@riscv_32::@entry
      // CHECK-NOT: @ex::@x86_64::@entry
      stream.cmd.dispatch @ex::@entry {
        rw %arg2[%c0 for %arg1] : !stream.resource<transient>{%arg1}
      }
    } => !stream.timepoint
    util.return %0 : !stream.timepoint
  }
}
