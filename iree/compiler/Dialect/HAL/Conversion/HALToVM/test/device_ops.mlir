// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK-LABEL: @device_allocator
func @device_allocator() -> !ireex.ref<!hal.allocator> {
  %0 = "test_hal.device"() : () -> !ireex.ref<!hal.device>
  // CHECK: %ref = vm.call @_hal.device.allocator(%0) : (!ireex.ref<!hal.device>) -> !ireex.ref<!hal.allocator>
  %allocator = hal.device.allocator %0 : !ireex.ref<!hal.allocator>
  return %allocator : !ireex.ref<!hal.allocator>
}
