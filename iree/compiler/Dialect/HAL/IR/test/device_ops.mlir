// Tests printing and parsing of hal.device ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @device_allocator
func @device_allocator() -> !ireex.ref<!hal.allocator> {
  %0 = "test_hal.device"() : () -> !ireex.ref<!hal.device>
  // CHECK: %allocator = hal.device.allocator %0 : !ireex.ref<!hal.allocator>
  %allocator = hal.device.allocator %0 : !ireex.ref<!hal.allocator>
  return %allocator : !ireex.ref<!hal.allocator>
}
