// RUN: iree-opt -allow-unregistered-dialect -split-input-file %s | iree-opt -allow-unregistered-dialect -split-input-file | IreeFileCheck %s

// CHECK-LABEL: byte_range.offset_length
"byte_range.offset_length"() {
  // CHECK: br = #hal.byte_range<123, 456>
  br = #hal.byte_range<123, 456>
} : () -> ()

// -----

// CHECK-LABEL: descriptor_set_layout_binding.basic
"descriptor_set_layout_binding.basic"() {
  // CHECK: dslb0 = #hal.descriptor_set_layout_binding<0, "StorageBuffer", RA>
  dslb0 = #hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read|MayAlias">,
  // CHECK: dslb1 = #hal.descriptor_set_layout_binding<0, "StorageBuffer", RA>
  dslb1 = #hal.descriptor_set_layout_binding<0, "StorageBuffer", RA>
} : () -> ()
