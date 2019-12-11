// Tests printing and parsing of hal.descriptor_set ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @descriptor_set_allocate
func @descriptor_set_allocate(%arg0 : !ireex.ref<!hal.device>, %arg1 : !ireex.ref<!hal.descriptor_set_layout>) {
  // CHECK: %descriptor_set = hal.descriptor_set.allocate %arg0, %arg1 : !ireex.ref<!hal.descriptor_set>
  %0 = hal.descriptor_set.allocate %arg0, %arg1 : !ireex.ref<!hal.descriptor_set>
  return
}

// -----

// CHECK-LABEL: @descriptor_set_make_binding
func @descriptor_set_make_binding() {
  %0 = "test_hal.buffer"() : () -> !ireex.ref<!hal.buffer>
  %1 = "test_hal.offset"() : () -> i32
  %2 = "test_hal.length"() : () -> i32
  // CHECK: %binding = hal.descriptor_set.make_binding binding=0, %0, %1, %2, "Read|Write" : tuple<i32, !ireex.ref<!hal.buffer>, i32, i32, i32>
  %3 = hal.descriptor_set.make_binding binding=0, %0, %1, %2, "Read|Write" : tuple<i32, !ireex.ref<!hal.buffer>, i32, i32, i32>
  return
}

// -----

// CHECK-LABEL: @descriptor_set_update
func @descriptor_set_update(%arg0 : !ireex.ref<!hal.device>, %arg1 : !ireex.ref<!hal.descriptor_set>) {
  %0 = "test_hal.buffer"() : () -> !ireex.ref<!hal.buffer>
  %1 = "test_hal.offset"() : () -> i32
  %2 = "test_hal.length"() : () -> i32
  // CHECK: [[B:%.+]] = hal.descriptor_set.make_binding
  %3 = hal.descriptor_set.make_binding binding=0, %0, %1, %2, "Read|Write" : tuple<i32, !ireex.ref<!hal.buffer>, i32, i32, i32>
  // CHECK-NEXT: hal.descriptor_set.update %arg0, %arg1, bindings={{\[}}[[B]]{{\]}}
  hal.descriptor_set.update %arg0, %arg1, bindings=[%3]
  return
}
