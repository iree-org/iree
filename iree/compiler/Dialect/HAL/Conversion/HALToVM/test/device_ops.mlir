// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK-LABEL: @device_allocator
func @device_allocator() -> !iree.ref<!hal.allocator> {
  %0 = "test_hal.device"() : () -> !iree.ref<!hal.device>
  // CHECK: %ref = vm.call @hal.device.allocator(%0) : (!iree.ref<!hal.device>) -> !iree.ref<!hal.allocator>
  %allocator = hal.device.allocator %0 : !iree.ref<!hal.allocator>
  return %allocator : !iree.ref<!hal.allocator>
}
