// RUN: iree-opt -split-input-file -iree-hal-materialize-resource-caches %s | IreeFileCheck %s

//      CHECK: hal.variable @_descriptor_set_layout_0 init(@_descriptor_set_layout_0_initializer) : !hal.descriptor_set_layout
// CHECK-NEXT: func @_descriptor_set_layout_0_initializer() -> !hal.descriptor_set_layout attributes {sym_visibility = "private"} {
// CHECK-NEXT:   %dev = hal.ex.shared_device : !hal.device
// CHECK-NEXT:   %descriptor_set_layout = hal.descriptor_set_layout.create %dev, "PushOnly", bindings = [#hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Write">] : !hal.descriptor_set_layout
// CHECK-NEXT:   return %descriptor_set_layout : !hal.descriptor_set_layout
// CHECK-NEXT: }

// CHECK-LABEL: @descriptorSetLayoutLookup
func @descriptorSetLayoutLookup(%arg0 : !hal.device) -> !hal.descriptor_set_layout {
  // CHECK-NEXT: %[[LAYOUT:.+]] = hal.variable.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
  %0 = hal.descriptor_set_layout.lookup %arg0, "PushOnly", bindings = [
    #hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">,
    #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Write">
  ] : !hal.descriptor_set_layout
  // CHECK-NEXT: return %[[LAYOUT]]
  return %0 : !hal.descriptor_set_layout
}

// -----

// CHECK: hal.variable @_descriptor_set_layout_0 init(@_descriptor_set_layout_0_initializer) : !hal.descriptor_set_layout

//      CHECK: hal.variable @_executable_layout_0 init(@_executable_layout_0_initializer) : !hal.executable_layout
// CHECK-NEXT: func @_executable_layout_0_initializer() -> !hal.executable_layout attributes {sym_visibility = "private"} {
// CHECK-NEXT:   %0 = hal.variable.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
// CHECK-NEXT:   %dev = hal.ex.shared_device : !hal.device
// CHECK-NEXT:   %executable_layout = hal.executable_layout.create %dev, set_layouts = [%0], push_constants = 0 : !hal.executable_layout
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
// CHECK-NEXT: func @_executable_layout_0_initializer() -> !hal.executable_layout attributes {sym_visibility = "private"} {
// CHECK-NEXT:   %0 = hal.variable.load @_descriptor_set_layout_0 : !hal.descriptor_set_layout
// CHECK-NEXT:   %1 = hal.variable.load @_descriptor_set_layout_1 : !hal.descriptor_set_layout
// CHECK-NEXT:   %dev = hal.ex.shared_device : !hal.device
// CHECK-NEXT:   %executable_layout = hal.executable_layout.create %dev, set_layouts = [%0, %1], push_constants = 0 : !hal.executable_layout
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
  %0 = hal.descriptor_set_layout.lookup %arg0, "PushOnly", bindings = [
    #hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">,
    #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Write">
  ] : !hal.descriptor_set_layout
  // CHECK-NEXT: return %[[LAYOUT]]
  return %0 : !hal.descriptor_set_layout
}

// -----

hal.executable @exe {
  hal.interface @interface {
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.entry_point @entry attributes {
    interface = @interface,
    ordinal = 0 : i32,
    signature = (tensor<4xf32>) -> tensor<4xf32>,
    workgroup_size = [32 : index, 1 : index, 1 : index]
  }
  hal.executable.entry_point @entry_alias attributes {
    interface = @interface,
    ordinal = 0 : i32,
    signature = (tensor<4xf32>) -> tensor<4xf32>,
    workgroup_size = [32 : index, 1 : index, 1 : index]
  }
  hal.executable.binary attributes {
    data = dense<[0, 1, 2, 3]> : vector<4xi8>,
    format = 1230128453 : i32
  }
  hal.executable.binary attributes {
    data = dense<[4, 5, 6, 7]> : vector<4xi8>,
    format = 1397773893 : i32
  }
}

//      CHECK: hal.variable @_executable_cache init(@_executable_cache_initializer) : !hal.executable_cache
// CHECK-NEXT: func @_executable_cache_initializer
//      CHECK: %[[CACHE:.+]] = hal.executable_cache.create %dev, identifier = "default" : !hal.executable_cache
// CHECK-NEXT: %[[LAYOUT:.+]] = hal.variable.load @_executable_layout_0 : !hal.executable_layout
// CHECK-NEXT: %[[EXE:.+]] = hal.executable_cache.prepare %[[CACHE]], layout = %[[LAYOUT]], caching_mode = "AliasProvidedData|AllowPersistentCaching|AllowOptimization", @exe : !hal.executable

// CHECK-LABEL: @exeLookup
func @exeLookup(%arg0 : !hal.device) -> !hal.executable {
  // CHECK: %[[EXE:.+]] = hal.variable.load @_executable_exe : !hal.executable
  %0 = hal.executable.lookup %arg0, @exe : !hal.executable
  // CHECK-NEXT: return %[[EXE]]
  return %0 : !hal.executable
}
