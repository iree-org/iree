// RUN: iree-opt -split-input-file %s | IreeFileCheck %s

#executable_target_format = #hal.executable.target<"backend", "format">

// CHECK-LABEL: @ex
hal.executable @ex {
  // CHECK: hal.executable.variant public @backend, target = #executable_target_format
  hal.executable.variant @backend, target = #executable_target_format {
    // CHECK-DAG: hal.executable.entry_point public @entry0 attributes {
    // CHECK-SAME:     interface = @interface
    // CHECK-SAME:     ordinal = 0 : index
    // CHECK-SAME:     workgroup_size = [4 : index, 1 : index, 1 : index]
    hal.executable.entry_point @entry0 attributes {
      interface = @interface,
      ordinal = 0 : index,
      workgroup_size = [4 : index, 1 : index, 1 : index]
    }
  }
  // CHECK-DAG: hal.interface public @interface
  hal.interface @interface {
    // CHECK-NEXT: hal.interface.binding public @s0b0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer"
    // CHECK-NEXT: hal.interface.binding public @s0b1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer"
  }
  // CHECK: hal.executable.binary
  hal.executable.binary @backend_binary attributes {
    // CHECK-SAME: data = dense<1> : vector<128xi8>,
    data = dense<1> : vector<128xi8>,
    // CHECK-SAME: format = "some_format"
    format = "some_format"
  }
}

// -----

#executable_target_format = #hal.executable.target<"backend", "format">

// CHECK-LABEL: @ex_with_workgroup_count_region
hal.executable @ex_with_workgroup_count_region {
  // CHECK: hal.executable.variant public @backend, target = #executable_target_format
  hal.executable.variant @backend, target = #executable_target_format {
    // CHECK-DAG: hal.executable.entry_point public @entry0 attributes {
    // CHECK-SAME:     interface = @interface
    // CHECK-SAME:     ordinal = 0 : index
    // CHECK-SAME:     workgroup_size = [4 : index, 1 : index, 1 : index]
    hal.executable.entry_point @entry0 attributes {
      interface = @interface,
      ordinal = 0 : index,
      workgroup_size = [4 : index, 1 : index, 1 : index]
    } {
    ^bb0(%arg0: index, %arg1: index, %arg2: index):
      hal.return %arg0, %arg1, %arg2 : index, index, index
    }
  }
  // CHECK-DAG: hal.interface public @interface
  hal.interface @interface {
    // CHECK-NEXT: hal.interface.binding public @s0b0, set=0, binding=0, type="StorageBuffer"
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer"
    // CHECK-NEXT: hal.interface.binding public @s0b1, set=0, binding=1, type="StorageBuffer"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer"
  }
  // CHECK: hal.executable.binary
  hal.executable.binary @backend_binary attributes {
    // CHECK-SAME: data = dense<1> : vector<128xi8>,
    data = dense<1> : vector<128xi8>,
    // CHECK-SAME: format = "some_format"
    format = "some_format"
  }
}

// -----

// CHECK-LABEL: @executable_create
// CHECK-SAME: %[[DEVICE:.+]]: !hal.device,
// CHECK-SAME: %[[LAYOUT0:.+]]: !hal.executable_layout,
// CHECK-SAME: %[[LAYOUT1:.+]]: !hal.executable_layout
func @executable_create(%device: !hal.device,
                        %layout0: !hal.executable_layout,
                        %layout1: !hal.executable_layout) {
  //      CHECK: = hal.executable.create
  // CHECK-SAME:     device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME:     target(@exe::@binary1)
  // CHECK-SAME:    layouts([%[[LAYOUT0]], %[[LAYOUT1]]]) : !hal.executable
  %0 = hal.executable.create device(%device : !hal.device)
                             target(@exe::@binary1)
                            layouts([%layout0, %layout1]) : !hal.executable
  return
}

// -----

// CHECK-LABEL: @executable_layout_create
// CHECK-SAME: %[[DEVICE:.+]]: !hal.device,
// CHECK-SAME: %[[LAYOUT0:.+]]: !hal.descriptor_set_layout,
// CHECK-SAME: %[[LAYOUT1:.+]]: !hal.descriptor_set_layout
func @executable_layout_create(%device: !hal.device,
                               %layout0: !hal.descriptor_set_layout,
                               %layout1: !hal.descriptor_set_layout) {
  // CHECK: hal.executable_layout.create
  // CHECK-SAME:          device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME:  push_constants(1)
  // CHECK-SAME:         layouts([%[[LAYOUT0]], %[[LAYOUT1]]]) : !hal.executable_layout
  %0 = hal.executable_layout.create device(%device : !hal.device)
                            push_constants(1)
                                   layouts([%layout0, %layout1]) : !hal.executable_layout
  return
}
