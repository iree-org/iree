// RUN: iree-opt --allow-unregistered-dialect --split-input-file --mlir-print-local-scope %s | FileCheck %s

// CHECK-LABEL: descriptor_set_layout_binding.basic
"descriptor_set_layout_binding.basic"() {
  // CHECK: dslb0 = #hal.descriptor_set.binding<0, uniform_buffer>
  dslb0 = #hal.descriptor_set.binding<0, uniform_buffer>,
  // CHECK: dslb1 = #hal.descriptor_set.binding<1, storage_buffer>
  dslb1 = #hal.descriptor_set.binding<1, storage_buffer>
} : () -> ()

// -----

// CHECK-LABEL: pipeline_layout.basic
"pipeline_layout.basic"() {
  // CHECK: layout0 = #hal.pipeline.layout<push_constants = 4, sets = [
  // CHECK-SAME: <0, bindings = [
  // CHECK-SAME:   <0, storage_buffer>
  // CHECK-SAME:   <1, storage_buffer>
  // CHECK-SAME: <1, bindings = [
  // CHECK-SAME:   <0, uniform_buffer>
  layout0 = #hal.pipeline.layout<push_constants = 4, sets = [
    #hal.descriptor_set.layout<0, bindings = [
      #hal.descriptor_set.binding<0, storage_buffer>,
      #hal.descriptor_set.binding<1, storage_buffer>
    ]>,
    #hal.descriptor_set.layout<1, bindings = [
      #hal.descriptor_set.binding<0, uniform_buffer>
    ]>
  ]>
} : () -> ()

// -----

"affinity.queue"() {
  // CHECK: any = #hal.affinity.queue<*>
  any = #hal.affinity.queue<*>,
  // CHECK: q0 = #hal.affinity.queue<[0]>
  q0 = #hal.affinity.queue<[0]>,
  // CHECK: q123 = #hal.affinity.queue<[1, 2, 3]>
  q123 = #hal.affinity.queue<[1, 2, 3]>
} : () -> ()
