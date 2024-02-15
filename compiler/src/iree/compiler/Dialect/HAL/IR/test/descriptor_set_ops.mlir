// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @descriptor_set_layout_create
// CHECK-SAME: (%[[DEVICE:.+]]: !hal.device)
util.func public @descriptor_set_layout_create(%device: !hal.device) {
  //      CHECK: = hal.descriptor_set_layout.create
  // CHECK-SAME:     device(%[[DEVICE]] : !hal.device)
  // CHECK-SAME:     flags("None")
  // CHECK-SAME:     bindings([
  // CHECK-SAME:       #hal.descriptor_set.binding<0, storage_buffer>,
  // CHECK-SAME:       #hal.descriptor_set.binding<1, storage_buffer>
  // CHECK-SAME:     ]) : !hal.descriptor_set_layout
  %0 = hal.descriptor_set_layout.create device(%device : !hal.device)
                                        flags("None")
                                        bindings([
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]) : !hal.descriptor_set_layout
  util.return
}
