// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: byte_range.offset_length
"byte_range.offset_length"() {
  // CHECK: br = #hal.byte_range<123, 456>
  br = #hal.byte_range<123, 456>
} : () -> ()

// -----

// CHECK-LABEL: descriptor_set_layout_binding.basic
"descriptor_set_layout_binding.basic"() {
  // CHECK: dslb = #hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read|MayAlias">
  dslb = #hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read|MayAlias">
} : () -> ()
