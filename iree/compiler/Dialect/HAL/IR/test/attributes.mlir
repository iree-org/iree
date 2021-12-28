// RUN: iree-opt -allow-unregistered-dialect -split-input-file -mlir-print-local-scope %s | IreeFileCheck %s

// CHECK-LABEL: descriptor_set_layout_binding.basic
"descriptor_set_layout_binding.basic"() {
  // CHECK: dslb0 = #hal.descriptor_set.binding<0, uniform_buffer>
  dslb0 = #hal.descriptor_set.binding<0, uniform_buffer>,
  // CHECK: dslb1 = #hal.descriptor_set.binding<1, storage_buffer>
  dslb1 = #hal.descriptor_set.binding<1, storage_buffer>
} : () -> ()

// -----

// CHECK-LABEL: executable_layout.basic
"executable_layout.basic"() {
  // CHECK: layout0 = #hal.executable.layout<push_constants = 4, sets = [
  // CHECK-SAME: #hal.descriptor_set.layout<0, bindings = [
  // CHECK-SAME:   #hal.descriptor_set.binding<0, storage_buffer>
  // CHECK-SAME:   #hal.descriptor_set.binding<1, storage_buffer>
  // CHECK-SAME: #hal.descriptor_set.layout<1, bindings = [
  // CHECK-SAME:   #hal.descriptor_set.binding<0, uniform_buffer>
  layout0 = #hal.executable.layout<push_constants = 4, sets = [
    #hal.descriptor_set.layout<0, bindings = [
      #hal.descriptor_set.binding<0, storage_buffer>,
      #hal.descriptor_set.binding<1, storage_buffer>
    ]>,
    #hal.descriptor_set.layout<1, bindings = [
      #hal.descriptor_set.binding<0, uniform_buffer>
    ]>
  ]>
} : () -> ()
