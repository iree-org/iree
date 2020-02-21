// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK-LABEL: @device_allocator
func @device_allocator(%arg0 : !hal.device) -> !hal.allocator {
  // CHECK: %ref = vm.call @hal.device.allocator(%arg0) : (!vm.ref<!hal.device>) -> !vm.ref<!hal.allocator>
  %allocator = hal.device.allocator %arg0 : !hal.allocator
  return %allocator : !hal.allocator
}
