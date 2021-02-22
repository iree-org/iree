// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @descriptor_set_layout_create
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
func @descriptor_set_layout_create(%device: !hal.device) {
  //      CHECK: = hal.descriptor_set_layout.create
  // CHECK-SAME:     device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME:     usage(PushOnly)
  // CHECK-SAME:     bindings([
  // CHECK-SAME:       #hal.descriptor_set_layout_binding<0, "StorageBuffer", R>,
  // CHECK-SAME:       #hal.descriptor_set_layout_binding<1, "StorageBuffer", W>
  // CHECK-SAME:     ]) : !hal.descriptor_set_layout
  %0 = hal.descriptor_set_layout.create device(%device : !hal.device)
                                        usage(PushOnly)
                                        bindings([
    #hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">,
    #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Write">
  ]) : !hal.descriptor_set_layout
  return
}
