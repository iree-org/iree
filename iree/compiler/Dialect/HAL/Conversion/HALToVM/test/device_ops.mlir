// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK-LABEL: @device_allocator
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>)
func @device_allocator(%device: !hal.device) -> !hal.allocator {
  // CHECK: %ref = vm.call @hal.device.allocator(%[[DEVICE]]) : (!vm.ref<!hal.device>) -> !vm.ref<!hal.allocator>
  %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
  return %allocator : !hal.allocator
}

// -----

// CHECK-LABEL: @device_query_i32
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>)
func @device_query_i32(%device: !hal.device) -> (i1, i32) {
  // CHECK: %[[KEY:.+]] = vm.const.ref.rodata @_utf8_foo_AFF8A909E08B2673 : !vm.ref<!iree.byte_buffer>
  // CHECK: %[[RET:.+]]:2 = vm.call @hal.device.query.i32(%[[DEVICE]], %[[KEY]]) : (!vm.ref<!hal.device>, !vm.ref<!iree.byte_buffer>) -> (i32, i32)
  %ok, %value = hal.device.query<%device : !hal.device> key("foo") : i1, i32
  // CHECK: return %[[RET]]#0, %[[RET]]#1
  return %ok, %value : i1, i32
}
