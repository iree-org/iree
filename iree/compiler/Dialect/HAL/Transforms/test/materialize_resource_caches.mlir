// RUN: iree-opt -split-input-file -iree-hal-materialize-resource-caches %s -iree-hal-target-backends=vmla | IreeFileCheck %s

//      CHECK: hal.variable @_descriptor_set_layout_0 init(@_descriptor_set_layout_0_initializer) : !hal.descriptor_set_layout
// CHECK-NEXT: func private @_descriptor_set_layout_0_initializer() -> !hal.descriptor_set_layout {
// CHECK-NEXT:   %dev = hal.ex.shared_device : !hal.device
// CHECK-NEXT:   %descriptor_set_layout = hal.descriptor_set_layout.create %dev, PushOnly, bindings = [#hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Write">] : !hal.descriptor_set_layout
// CHECK-NEXT:   return %descriptor_set_layout : !hal.descriptor_set_layout
// CHECK-NEXT: }

// CHECK-LABEL: @descriptorSetLayoutLookup
func @descriptorSetLayoutLookup(%arg0 : !hal.device) -> !hal.descriptor_set_layout {
  // CHECK-NEXT: %[[LAYOUT:.+]] = hal.variable.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
  %0 = hal.descriptor_set_layout.lookup %arg0, PushOnly, bindings = [
    #hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">,
    #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Write">
  ] : !hal.descriptor_set_layout
  // CHECK-NEXT: return %[[LAYOUT]]
  return %0 : !hal.descriptor_set_layout
}

// -----

// CHECK: hal.variable @_descriptor_set_layout_0 init(@_descriptor_set_layout_0_initializer) : !hal.descriptor_set_layout

//      CHECK: hal.variable @_executable_layout_0 init(@_executable_layout_0_initializer) : !hal.executable_layout
// CHECK-NEXT: func private @_executable_layout_0_initializer() -> !hal.executable_layout {
// CHECK-NEXT:   %0 = hal.variable.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
// CHECK-NEXT:   %dev = hal.ex.shared_device : !hal.device
// CHECK-NEXT:   %executable_layout = hal.executable_layout.create %dev, push_constants = 0, set_layouts = [%0] : !hal.executable_layout
// CHECK-NEXT:   return %executable_layout : !hal.executable_layout
// CHECK-NEXT: }

// CHECK-LABEL: @exeLayoutLookup
func @exeLayoutLookup(%arg0 : !hal.device) -> !hal.executable_layout {
  // CHECK: %[[LAYOUT:.+]] = hal.variable.load @_executable_layout_0 : !hal.executable_layout
  %0 = hal.executable_layout.lookup %arg0, set_layouts = [
    [
      #hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">,
      #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Write">
    ]
  ] : !hal.executable_layout
  // CHECK-NEXT: return %[[LAYOUT]]
  return %0 : !hal.executable_layout
}

// -----

// CHECK: hal.variable @_descriptor_set_layout_0
// CHECK: hal.variable @_descriptor_set_layout_1

//      CHECK: hal.variable @_executable_layout_0 init(@_executable_layout_0_initializer) : !hal.executable_layout
// CHECK-NEXT: func private @_executable_layout_0_initializer() -> !hal.executable_layout {
// CHECK-NEXT:   %0 = hal.variable.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
// CHECK-NEXT:   %1 = hal.variable.load @_descriptor_set_layout_1 : !hal.descriptor_set_layout
// CHECK-NEXT:   %dev = hal.ex.shared_device : !hal.device
// CHECK-NEXT:   %executable_layout = hal.executable_layout.create %dev, push_constants = 0, set_layouts = [%0, %1] : !hal.executable_layout
// CHECK-NEXT:   return %executable_layout : !hal.executable_layout
// CHECK-NEXT: }

// CHECK-LABEL: @sharedLayoutLookup
func @sharedLayoutLookup(%arg0 : !hal.device) -> !hal.executable_layout {
  // CHECK: %[[LAYOUT:.+]] = hal.variable.load @_executable_layout_0 : !hal.executable_layout
  %0 = hal.executable_layout.lookup %arg0, set_layouts = [
    [
      #hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">,
      #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Write">
    ],
    [
      #hal.descriptor_set_layout_binding<0, "UniformBuffer", "Read">,
      #hal.descriptor_set_layout_binding<1, "UniformBuffer", "Write">
    ]
  ] : !hal.executable_layout
  // CHECK-NEXT: return %[[LAYOUT]]
  return %0 : !hal.executable_layout
}

// CHECK: @otherDescriptorSetLayoutLookup
func @otherDescriptorSetLayoutLookup(%arg0 : !hal.device) -> !hal.descriptor_set_layout {
  // CHECK: %[[LAYOUT:.+]] = hal.variable.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
  %0 = hal.descriptor_set_layout.lookup %arg0, PushOnly, bindings = [
    #hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">,
    #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Write">
  ] : !hal.descriptor_set_layout
  // CHECK-NEXT: return %[[LAYOUT]]
  return %0 : !hal.descriptor_set_layout
}

// -----

// TODO(scotttodd): Test without depending on a specific HAL target? Or move to HAL/Target/*/test/?
//   - If there is no matching hal.executable.target then the executable will not be cached
hal.executable @exe {
  hal.interface @interface0 {
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.interface @interface1 {
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b2, set=0, binding=2, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.target @vmla, filter="vmla" {
    hal.executable.entry_point @entry0 attributes {
      interface = @interface0,
      ordinal = 0 : i32,
      signature = (tensor<4xf32>) -> tensor<4xf32>,
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
    hal.executable.entry_point @entry0_alias attributes {
      interface = @interface0,
      ordinal = 0 : i32,
      signature = (tensor<4xf32>) -> tensor<4xf32>,
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
    hal.executable.entry_point @entry1 attributes {
      interface = @interface1,
      ordinal = 1 : i32,
      signature = (tensor<4xf32>, tensor<8xf32>) -> tensor<4xf32>,
      workgroup_size = [32 : index, 1 : index, 1 : index]
    }
  }
}

// CHECK-DAG: hal.variable @_descriptor_set_layout_0
// CHECK-DAG: hal.variable @_executable_layout_0
// CHECK-DAG: hal.variable @_descriptor_set_layout_1
// CHECK-DAG: hal.variable @_executable_layout_1

// CHECK: hal.variable @_executable_exe init(@_executable_exe_initializer) : !hal.executable
// CHECK: func private @_executable_exe_initializer() -> !hal.executable {
// CHECK:   %[[IN_DEV:.+]] = hal.ex.shared_device : !hal.device
// CHECK:   %[[RET:.+]] = hal.device.switch(%[[IN_DEV]] : !hal.device) -> !hal.executable
// CHECK:   #hal.device.match.id<"vmla">(%[[DEV:.+]] = %[[IN_DEV]] : !hal.device) {
// CHECK:     %[[LAYOUT0:.+]] = hal.variable.load @_executable_layout_0 : !hal.executable_layout
// CHECK:     %[[LAYOUT0_2:.+]] = hal.variable.load @_executable_layout_0 : !hal.executable_layout
// CHECK:     %[[LAYOUT1:.+]] = hal.variable.load @_executable_layout_1 : !hal.executable_layout
// CHECK:     %[[EXE:.+]] = hal.executable.create %[[DEV]], @exe::@vmla, layouts = [%[[LAYOUT0]], %[[LAYOUT0_2]], %[[LAYOUT1]]] : !hal.executable
// CHECK:     hal.return %[[EXE]] : !hal.executable
// CHECK:   },
// CHECK:   #hal.match.always() {
// CHECK:     %[[NULL:.+]] = iree.null : !hal.executable
// CHECK:     hal.return %[[NULL]] : !hal.executable
// CHECK:   }
// CHECK:   return %[[RET]] : !hal.executable
// CHECK: }

// CHECK-LABEL: @exeLookup
func @exeLookup(%arg0 : !hal.device) -> !hal.executable {
  // CHECK: %[[EXE:.+]] = hal.variable.load @_executable_exe : !hal.executable
  %0 = hal.executable.lookup %arg0, @exe : !hal.executable
  // CHECK-NEXT: return %[[EXE]]
  return %0 : !hal.executable
}
