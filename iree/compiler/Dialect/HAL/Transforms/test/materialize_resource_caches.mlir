// RUN: iree-opt -split-input-file -iree-hal-materialize-resource-caches %s | FileCheck %s

//      CHECK: util.global private @_descriptor_set_layout_0 : !hal.descriptor_set_layout
// CHECK-NEXT: util.initializer {
// CHECK-NEXT:   %device = hal.ex.shared_device : !hal.device
// CHECK-NEXT:   %descriptor_set_layout = hal.descriptor_set_layout.create
// CHECK-SAME:     device(%device : !hal.device)
// CHECK-SAME:     usage(push_only)
// CHECK-SAME:     bindings([
// CHECK-SAME:       #hal.descriptor_set.binding<0, storage_buffer>,
// CHECK-SAME:       #hal.descriptor_set.binding<1, storage_buffer>
// CHECK-SAME:     ]) : !hal.descriptor_set_layout
// CHECK-NEXT:   util.global.store %descriptor_set_layout, @_descriptor_set_layout_0 : !hal.descriptor_set_layout

// CHECK-LABEL: @descriptorSetLayoutLookup
func @descriptorSetLayoutLookup(%device : !hal.device) -> !hal.descriptor_set_layout {
  // CHECK-NEXT: %[[LAYOUT:.+]] = util.global.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
  %0 = hal.descriptor_set_layout.lookup device(%device : !hal.device)
                                        usage(push_only)
                                        bindings([
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]) : !hal.descriptor_set_layout
  // CHECK-NEXT: return %[[LAYOUT]]
  return %0 : !hal.descriptor_set_layout
}

// -----

// CHECK: util.global private @_descriptor_set_layout_0 : !hal.descriptor_set_layout

//      CHECK: util.global private @_executable_layout_0 : !hal.executable_layout
// CHECK-NEXT: util.initializer {
// CHECK-NEXT:   %[[SET0:.+]] = util.global.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
// CHECK-NEXT:   %device = hal.ex.shared_device : !hal.device
// CHECK-NEXT:   %executable_layout = hal.executable_layout.create
// CHECK-SAME:     device(%device : !hal.device)
// CHECK-SAME:     push_constants(1)
// CHECK-SAME:     layouts([%[[SET0]]]) : !hal.executable_layout
// CHECK-NEXT:   util.global.store %executable_layout, @_executable_layout_0 : !hal.executable_layout

// CHECK-LABEL: @exeLayoutLookup
func @exeLayoutLookup(%device : !hal.device) -> !hal.executable_layout {
  // CHECK: %[[LAYOUT:.+]] = util.global.load @_executable_layout_0 : !hal.executable_layout
  %0 = hal.executable_layout.lookup device(%device : !hal.device)
                                    layout(#hal.executable.layout<push_constants = 1, sets = [
    #hal.descriptor_set.layout<0, bindings = [
      #hal.descriptor_set.binding<0, storage_buffer>,
      #hal.descriptor_set.binding<1, storage_buffer>
    ]>
  ]>) : !hal.executable_layout
  // CHECK-NEXT: return %[[LAYOUT]]
  return %0 : !hal.executable_layout
}

// -----

// CHECK: util.global private @_descriptor_set_layout_0
// CHECK: util.global private @_descriptor_set_layout_1

//      CHECK: util.global private @_executable_layout_0 : !hal.executable_layout
// CHECK-NEXT: util.initializer {
// CHECK-NEXT:   %[[SET0:.+]] = util.global.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
// CHECK-NEXT:   %[[SET1:.+]] = util.global.load @_descriptor_set_layout_1 : !hal.descriptor_set_layout
// CHECK-NEXT:   %device = hal.ex.shared_device : !hal.device
// CHECK-NEXT:   %executable_layout = hal.executable_layout.create
// CHECK-SAME:     device(%device : !hal.device)
// CHECK-SAME:     push_constants(1)
// CHECK-SAME:     layouts([%[[SET0]], %[[SET1]]]) : !hal.executable_layout
// CHECK-NEXT:   util.global.store %executable_layout, @_executable_layout_0 : !hal.executable_layout

// CHECK-LABEL: @sharedLayoutLookup
func @sharedLayoutLookup(%device : !hal.device) -> !hal.executable_layout {
  // CHECK: %[[LAYOUT:.+]] = util.global.load @_executable_layout_0 : !hal.executable_layout
  %0 = hal.executable_layout.lookup device(%device : !hal.device)
                                    layout(#hal.executable.layout<push_constants = 1, sets = [
    #hal.descriptor_set.layout<0, bindings = [
      #hal.descriptor_set.binding<0, storage_buffer>,
      #hal.descriptor_set.binding<1, storage_buffer>
    ]>,
    #hal.descriptor_set.layout<1, bindings = [
      #hal.descriptor_set.binding<0, uniform_buffer>,
      #hal.descriptor_set.binding<1, uniform_buffer>
    ]>
  ]>) : !hal.executable_layout
  // CHECK-NEXT: return %[[LAYOUT]]
  return %0 : !hal.executable_layout
}

// CHECK: @otherDescriptorSetLayoutLookup
func @otherDescriptorSetLayoutLookup(%device : !hal.device) -> !hal.descriptor_set_layout {
  // CHECK: %[[LAYOUT:.+]] = util.global.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
  %0 = hal.descriptor_set_layout.lookup device(%device : !hal.device)
                                        usage(push_only)
                                        bindings([
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]) : !hal.descriptor_set_layout
  // CHECK-NEXT: return %[[LAYOUT]]
  return %0 : !hal.descriptor_set_layout
}

// -----

#executable_layout_0 = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
#executable_layout_1 = #hal.executable.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

module attributes {hal.device.targets = [#hal.device.target<"cpu">]} {

// TODO(scotttodd): Test without depending on a specific HAL target? Or move to HAL/Target/*/test/?
//   - If there is no matching hal.executable.variant then the executable will not be cached
hal.executable @exe {
  hal.executable.variant @vmvx, target = <"vmvx", "vmvx-bytecode-fb"> {
    hal.executable.entry_point @entry0 ordinal(0) layout(#executable_layout_0) attributes {
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
    hal.executable.entry_point @entry0_alias ordinal(0) layout(#executable_layout_0) attributes {
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
    hal.executable.entry_point @entry1 ordinal(1) layout(#executable_layout_1) attributes {
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
  }
}

// CHECK-DAG: util.global private @_descriptor_set_layout_0
// CHECK-DAG: util.global private @_executable_layout_0
// CHECK-DAG: util.global private @_descriptor_set_layout_1
// CHECK-DAG: util.global private @_executable_layout_1

// CHECK: util.global private @_executable_exe : !hal.executable
// CHECK-NEXT: util.initializer {
// CHECK:   %[[DEV:.+]] = hal.ex.shared_device : !hal.device
// CHECK:   %[[RET:.+]] = hal.device.switch<%[[DEV]] : !hal.device> -> !hal.executable
// CHECK:   #hal.device.match.executable.format<"vmvx-bytecode-fb"> {
// CHECK:     %[[LAYOUT0:.+]] = util.global.load @_executable_layout_0 : !hal.executable_layout
// CHECK:     %[[LAYOUT0_2:.+]] = util.global.load @_executable_layout_0 : !hal.executable_layout
// CHECK:     %[[LAYOUT1:.+]] = util.global.load @_executable_layout_1 : !hal.executable_layout
// CHECK:     %[[EXE:.+]] = hal.executable.create
// CHECK-SAME:  device(%[[DEV]] : !hal.device)
// CHECK-SAME:  target(@exe::@vmvx)
// CHECK-SAME:  layouts([%[[LAYOUT0]], %[[LAYOUT0_2]], %[[LAYOUT1]]])
// CHECK-SAME:  : !hal.executable
// CHECK:     hal.return %[[EXE]] : !hal.executable
// CHECK:   },
// CHECK:   #hal.match.always {
// CHECK:     %[[NULL:.+]] = util.null : !hal.executable
// CHECK:     hal.return %[[NULL]] : !hal.executable
// CHECK:   }
// CHECK:   util.global.store %[[RET]], @_executable_exe : !hal.executable

// CHECK-LABEL: @exeLookup
func @exeLookup(%device : !hal.device) -> !hal.executable {
  // CHECK: %[[EXE:.+]] = util.global.load @_executable_exe : !hal.executable
  %0 = hal.executable.lookup device(%device : !hal.device)
                             executable(@exe) : !hal.executable
  // CHECK-NEXT: return %[[EXE]]
  return %0 : !hal.executable
}

}
