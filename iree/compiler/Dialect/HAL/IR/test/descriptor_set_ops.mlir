// Tests printing and parsing of hal.descriptor_set ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @descriptor_set_make_binding
func @descriptor_set_make_binding() {
  %0 = "test_hal.buffer"() : () -> !iree.ref<!hal.buffer>
  %1 = "test_hal.offset"() : () -> i32
  %2 = "test_hal.length"() : () -> i32
  // CHECK: %binding = hal.descriptor_set.make_binding binding = 0, %0, %1, %2 : tuple<i32, !iree.ref<!hal.buffer>, i32, i32>
  %3 = hal.descriptor_set.make_binding binding = 0, %0, %1, %2 : tuple<i32, !iree.ref<!hal.buffer>, i32, i32>
  return
}

// -----

// CHECK-LABEL: @descriptor_set_create
func @descriptor_set_create(%arg0 : !iree.ref<!hal.device>, %arg1 : !iree.ref<!hal.descriptor_set_layout>) {
  %0 = "test_hal.buffer"() : () -> !iree.ref<!hal.buffer>
  %1 = "test_hal.offset"() : () -> i32
  %2 = "test_hal.length"() : () -> i32
  // CHECK: %binding = hal.descriptor_set.make_binding binding = 0, %0, %1, %2 : tuple<i32, !iree.ref<!hal.buffer>, i32, i32>
  %binding = hal.descriptor_set.make_binding binding = 0, %0, %1, %2 : tuple<i32, !iree.ref<!hal.buffer>, i32, i32>
  // CHECK: %descriptor_set = hal.descriptor_set.create %arg0, %arg1, bindings = [%binding, %binding] : !iree.ref<!hal.descriptor_set>
  %descriptor_set = hal.descriptor_set.create %arg0, %arg1, bindings = [%binding, %binding] : !iree.ref<!hal.descriptor_set>
  return
}
