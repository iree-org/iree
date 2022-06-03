// RUN: iree-opt --split-input-file --iree-hal-materialize-interfaces %s | FileCheck %s

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
