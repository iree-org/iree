// Tests printing and parsing of hal.descriptor_set ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @descriptor_set_layout_create
func @descriptor_set_layout_create(%arg0 : !hal.device) {
  // CHECK: hal.descriptor_set_layout.create %arg0, "PushOnly", bindings = [#hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">, #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Write">] : !hal.descriptor_set_layout
  %descriptor_set_layout = hal.descriptor_set_layout.create %arg0, "PushOnly", bindings = [
    #hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read">,
    #hal.descriptor_set_layout_binding<1, "StorageBuffer", "Write">
  ] : !hal.descriptor_set_layout
  return
}
