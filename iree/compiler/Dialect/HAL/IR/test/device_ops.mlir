// Tests printing and parsing of hal.device ops.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @device_allocator
func @device_allocator() -> !hal.allocator {
  %0 = "test_hal.device"() : () -> !hal.device
  // CHECK: %allocator = hal.device.allocator %0 : !hal.allocator
  %allocator = hal.device.allocator %0 : !hal.allocator
  return %allocator : !hal.allocator
}
