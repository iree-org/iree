// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

"some.foo"() {
  // CHECK: dslb = #hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read|MayAlias">
  dslb = #hal.descriptor_set_layout_binding<0, "StorageBuffer", "Read|MayAlias">
} : () -> ()
