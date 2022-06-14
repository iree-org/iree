// RUN: iree-opt --split-input-file --iree-hal-materialize-interfaces %s | FileCheck %s

// Tests an executable with a workgroup count region specified.

module attributes {hal.device.targets = [
  #hal.device.target<"cpu", {
    executable_targets = [
      #hal.executable.target<"llvm", "embedded-elf-arm_64">,
      #hal.executable.target<"llvm", "embedded-elf-x86_64">
    ]
  }>
]} {

// CHECK: hal.executable private @ex_workgroups
// CHECK:   hal.executable.variant public @embedded_elf_arm_64, target = #executable_target_embedded_elf_arm_64
// CHECK:     hal.executable.export public @entry ordinal(0) layout(#executable_layout) {
// CHECK-NEXT: ^bb0(%[[DEVICE:.+]]: !hal.device, %[[ARG0:.+]]: index, %[[ARG1:.+]]: index):
// CHECK-NEXT:   hal.return %[[ARG0]], %[[ARG1]], %[[ARG0]] : index, index, index
// CHECK-NEXT: }
// CHECK:     builtin.module
// CHECK-NEXT:  func.func @entry()
// CHECK:   hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64
// CHECK:     hal.executable.export public @entry ordinal(0) layout(#executable_layout) {
// CHECK-NEXT: ^bb0(%[[DEVICE:.+]]: !hal.device, %[[ARG0:.+]]: index, %[[ARG1:.+]]: index):
// CHECK-NEXT:   hal.return %[[ARG0]], %[[ARG1]], %[[ARG0]] : index, index, index
// CHECK-NEXT: }
// CHECK:     builtin.module
// CHECK-NEXT:  func.func @entry()

stream.executable private @ex_workgroups {
  stream.executable.export public @entry workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
    stream.return %arg0, %arg1, %arg0 : index, index, index
  }
  builtin.module {
    func.func @entry() {
      return
    }
  }
}

// CHECK-LABEL: @main
func.func @main(%arg0: !stream.resource<external>, %arg1: index) -> !stream.resource<external> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %0 = stream.async.dispatch @ex_workgroups::@entry[%c1, %c2](%arg0, %c4) : (!stream.resource<external>{%arg1}, index) -> %arg0{%arg1}
  return %0 : !stream.resource<external>
}

}

// -----

// Tests an already-specified executable source op is expanded into the variants
// specified by the target configuration. These source executables may come from
// hand-authored code or other dialects that perform interface assignment
// themselves.

module attributes {hal.device.targets = [
  #hal.device.target<"cpu", {
    executable_targets = [
      #hal.executable.target<"llvm", "embedded-elf-arm_64">,
      #hal.executable.target<"llvm", "embedded-elf-x86_64">
    ]
  }>
]} {

hal.executable.source public @ex {
  hal.executable.export public @entry layout(#hal.executable.layout<push_constants = 1, sets = [
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
      %s0b0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%const0) alignment(32) : !flow.dispatch.tensor<readonly:4xf32>
      %s1b0 = hal.interface.binding.subspan set(1) binding(0) type(storage_buffer) offset(%const1) alignment(32) : !flow.dispatch.tensor<readonly:4xf32>
      %s1b1 = hal.interface.binding.subspan set(1) binding(1) type(storage_buffer) alignment(16) : !flow.dispatch.tensor<writeonly:4xf32>
      %workgroup_size_x = hal.interface.workgroup.size[0] : index
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_count_x = hal.interface.workgroup.count[0] : index
      return
    }
  }
}

// CHECK: hal.executable public @ex
// CHECK:   hal.executable.variant public @embedded_elf_arm_64, target = #executable_target_embedded_elf_arm_64
// CHECK:     hal.executable.export public @entry layout(#executable_layout)
// CHECK:     builtin.module
// CHECK-NEXT:  func.func @entry()
// CHECK:   hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64
// CHECK:     hal.executable.export public @entry layout(#executable_layout)
// CHECK:     builtin.module
// CHECK-NEXT:  func.func @entry()

// TODO(benvanik): test fixup of stream ops when attrs to specify the
// layout bindings are implemented.

}
