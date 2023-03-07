// RUN: iree-opt --split-input-file --iree-hal-materialize-interfaces %s | FileCheck %s

// Tests an executable with a workgroup count region specified.

module attributes {hal.device.targets = [
  #hal.device.target<"llvm-cpu", {
    executable_targets = [
      #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64">,
      #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
    ]
  }>
]} {

  // CHECK: #pipeline_layout = #hal.pipeline.layout<
  // CHECK-SAME: push_constants = 1
  // CHECK-SAME: sets = [
  // CHECK-SAME:   <0, bindings = [
  // CHECK-SAME:     <0, storage_buffer, ReadOnly>
  // CHECK-SAME:     <1, storage_buffer, ReadOnly>
  // CHECK-SAME:     <2, storage_buffer>

  // CHECK: hal.executable private @ex_workgroups
  // CHECK:   hal.executable.variant public @embedded_elf_arm_64, target = #executable_target_embedded_elf_arm_64
  // CHECK:     hal.executable.export public @entry ordinal(0) layout(#pipeline_layout) {
  // CHECK-NEXT: ^bb0(%[[DEVICE:.+]]: !hal.device, %[[ARG0:.+]]: index, %[[ARG1:.+]]: index):
  // CHECK-NEXT:   hal.return %[[ARG0]], %[[ARG1]], %[[ARG0]] : index, index, index
  // CHECK-NEXT: }
  // CHECK:     builtin.module
  // CHECK-NEXT:  func.func private @extern_func()
  // CHECK-NEXT:  func.func @entry
  // CHECK:   hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64
  // CHECK:     hal.executable.export public @entry ordinal(0) layout(#pipeline_layout) {
  // CHECK-NEXT: ^bb0(%[[DEVICE:.+]]: !hal.device, %[[ARG0:.+]]: index, %[[ARG1:.+]]: index):
  // CHECK-NEXT:   hal.return %[[ARG0]], %[[ARG1]], %[[ARG0]] : index, index, index
  // CHECK-NEXT: }
  // CHECK:     builtin.module
  // CHECK-NEXT:  func.func private @extern_func()
  // CHECK-NEXT:  func.func @entry

  stream.executable private @ex_workgroups {
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
  func.func @main(%arg0: !stream.resource<constant>, %arg1: !stream.resource<transient>, %arg2: index, %arg3: i32) -> !stream.resource<transient> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = stream.resource.alloc uninitialized : !stream.resource<transient>{%arg2}
    %1 = stream.cmd.execute with(%arg0 as %arg4: !stream.resource<constant>{%arg2}, %arg1 as %arg5: !stream.resource<transient>{%arg2}, %0 as %arg6: !stream.resource<transient>{%arg2}) {
      // CHECK: stream.cmd.dispatch {@ex_workgroups::@embedded_elf_arm_64::@entry, @ex_workgroups::@embedded_elf_x86_64::@entry}
      // CHECK: attributes {
      // CHECK-SAME: hal.interface.bindings = [
      // CHECK-SAME:   #hal.interface.binding<0, 0>,
      // CHECK-SAME:   #hal.interface.binding<0, 1>,
      // CHECK-SAME:   #hal.interface.binding<0, 2>
      stream.cmd.dispatch @ex_workgroups::@entry[%c1, %c2](%arg3 : i32) {
        ro %arg4[%c0 for %arg2] : !stream.resource<constant>{%arg2},
        ro %arg5[%c0 for %arg2] : !stream.resource<transient>{%arg2},
        wo %arg6[%c0 for %arg2] : !stream.resource<transient>{%arg2}
      }
    } => !stream.timepoint
    %2 = stream.timepoint.await %1 => %0 : !stream.resource<transient>{%arg2}
    return %2 : !stream.resource<transient>
  }
}

// -----

// Tests an already-specified executable source op is expanded into the variants
// specified by the target configuration. These source executables may come from
// hand-authored code or other dialects that perform interface assignment
// themselves.

module attributes {hal.device.targets = [
  #hal.device.target<"llvm-cpu", {
    executable_targets = [
      #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64">,
      #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64">
    ]
  }>
]} {

hal.executable.source public @ex {
  hal.executable.export public @entry layout(#hal.pipeline.layout<push_constants = 1, sets = [
    #hal.descriptor_set.layout<0, bindings = [
      #hal.descriptor_set.binding<0, storage_buffer>
    ]>,
    #hal.descriptor_set.layout<1, bindings = [
      #hal.descriptor_set.binding<0, storage_buffer>,
      #hal.descriptor_set.binding<1, storage_buffer>
    ]>
  ]>)
  builtin.module {
    func.func @entry() {
      %const0 = hal.interface.constant.load[0] : index
      %const1 = hal.interface.constant.load[1] : index
      %s0b0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%const0) : !flow.dispatch.tensor<readonly:tensor<4xf32>>
      %s1b0 = hal.interface.binding.subspan set(1) binding(0) type(storage_buffer) alignment(32) offset(%const1) : !flow.dispatch.tensor<readonly:tensor<4xf32>>
      %s1b1 = hal.interface.binding.subspan set(1) binding(1) type(storage_buffer) alignment(16) : !flow.dispatch.tensor<writeonly:tensor<4xf32>>
      %workgroup_size_x = hal.interface.workgroup.size[0] : index
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_count_x = hal.interface.workgroup.count[0] : index
      return
    }
  }
}

// CHECK: hal.executable public @ex
// CHECK:   hal.executable.variant public @embedded_elf_arm_64, target = #executable_target_embedded_elf_arm_64
// CHECK:     hal.executable.export public @entry layout(#pipeline_layout)
// CHECK:     builtin.module
// CHECK-NEXT:  func.func @entry()
// CHECK:   hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64
// CHECK:     hal.executable.export public @entry layout(#pipeline_layout)
// CHECK:     builtin.module
// CHECK-NEXT:  func.func @entry()

// TODO(benvanik): test fixup of stream ops when attrs to specify the
// layout bindings are implemented.

}
