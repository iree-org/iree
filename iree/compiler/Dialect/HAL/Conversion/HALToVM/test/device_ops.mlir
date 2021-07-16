// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm -canonicalize %s | IreeFileCheck %s

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
  // CHECK-DAG: %[[NS:.+]] = vm.rodata.inline "_utf8_sys_
  // CHECK-DAG: %[[KEY:.+]] = vm.rodata.inline "_utf8_foo_
  // CHECK: %[[RET:.+]]:2 = vm.call @hal.device.query.i32(%[[DEVICE]], %[[NS]], %[[KEY]]) : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer) -> (i32, i32)
  %ok, %value = hal.device.query<%device : !hal.device> key("sys" :: "foo") : i1, i32
  // CHECK: return %[[RET]]#0, %[[RET]]#1
  return %ok, %value : i1, i32
}

// -----

// CHECK-LABEL: @device_query_i32_default
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>)
func @device_query_i32_default(%device: !hal.device) -> i32 {
  // CHECK-DAG: %[[NS:.+]] = vm.rodata.inline "_utf8_sys_
  // CHECK-DAG: %[[KEY:.+]] = vm.rodata.inline "_utf8_foo_
  // CHECK: %[[RET:.+]]:2 = vm.call @hal.device.query.i32(%[[DEVICE]], %[[NS]], %[[KEY]]) : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer) -> (i32, i32)
  %ok, %value = hal.device.query<%device : !hal.device> key("sys" :: "foo") : i1, i32 = 123 : i32
  // CHECK: %[[OUT:.+]] = vm.select.i32 %[[RET]]#0, %[[RET]]#1, %c123 : i32
  // CHECK: return %[[OUT]]
  return %value : i32
}

// -----

// CHECK-LABEL: @device_query_i1
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>)
func @device_query_i1(%device: !hal.device) -> (i1, i1) {
  // CHECK-DAG: %[[NS:.+]] = vm.rodata.inline "_utf8_sys_
  // CHECK-DAG: %[[KEY:.+]] = vm.rodata.inline "_utf8_foo_
  // CHECK: %[[RET:.+]]:2 = vm.call @hal.device.query.i32(%[[DEVICE]], %[[NS]], %[[KEY]]) : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer) -> (i32, i32)
  %ok, %value = hal.device.query<%device : !hal.device> key("sys" :: "foo") : i1, i1
  // CHECK: %[[I1:.+]] = vm.and.i32 %[[RET]]#1, %c1 : i32
  // CHECK: return %[[RET]]#0, %[[I1]]
  return %ok, %value : i1, i1
}

// -----

// CHECK-LABEL: @device_query_i1_default
// CHECK-SAME: (%[[DEVICE:.+]]: !vm.ref<!hal.device>)
func @device_query_i1_default(%device: !hal.device) -> i1 {
  // CHECK-DAG: %[[NS:.+]] = vm.rodata.inline "_utf8_sys_
  // CHECK-DAG: %[[KEY:.+]] = vm.rodata.inline "_utf8_foo_
  // CHECK: %[[RET:.+]]:2 = vm.call @hal.device.query.i32(%[[DEVICE]], %[[NS]], %[[KEY]]) : (!vm.ref<!hal.device>, !vm.buffer, !vm.buffer) -> (i32, i32)
  %ok, %value = hal.device.query<%device : !hal.device> key("sys" :: "foo") : i1, i1 = 1 : i1
  // CHECK: %[[I1:.+]] = vm.and.i32 %[[RET]]#1, %c1 : i32
  // CHECK: %[[OUT:.+]] = vm.select.i32 %[[RET]]#0, %[[I1]], %c1
  // CHECK: return %[[OUT]]
  return %value : i1
}
