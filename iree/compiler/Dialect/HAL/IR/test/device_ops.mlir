// Tests printing and parsing of hal.device ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @device_allocator
func @device_allocator() -> !iree.ref<!hal.allocator> {
  %0 = "test_hal.device"() : () -> !iree.ref<!hal.device>
  // CHECK: %allocator = hal.device.allocator %0 : !iree.ref<!hal.allocator>
  %allocator = hal.device.allocator %0 : !iree.ref<!hal.allocator>
  return %allocator : !iree.ref<!hal.allocator>
}
