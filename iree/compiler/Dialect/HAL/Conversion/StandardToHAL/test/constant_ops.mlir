// RUN: iree-opt -split-input-file -iree-convert-to-hal %s | IreeFileCheck %s

// CHECK-LABEL: @constantTensor
func @constantTensor() {
  //      CHECK: %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
  //      CHECK: %cbuffer = hal.allocator.constant<%allocator : !hal.allocator>
  // CHECK-SAME:   type("HostVisible|DeviceVisible|DeviceLocal")
  // CHECK-SAME:   usage("Constant|Transfer|Mapping|Dispatch")
  // CHECK-SAME:   : !hal.buffer
  // CHECK-SAME:   = dense<[1, 2]> : tensor<2xi32>
  %0 = constant dense<[1, 2]> : tensor<2xi32>
  return
}

// -----

// CHECK-LABEL: @constantTensor1
func @constantTensor1() {
  //      CHECK: %allocator = hal.device.allocator<%device : !hal.device> : !hal.allocator
  //      CHECK: %cbuffer = hal.allocator.constant<%allocator : !hal.allocator>
  // CHECK-SAME:   type("HostVisible|DeviceVisible|DeviceLocal")
  // CHECK-SAME:   usage("Constant|Transfer|Mapping|Dispatch")
  // CHECK-SAME:   : !hal.buffer
  // CHECK-SAME:   = dense<[1, 0]> : tensor<2xi8>
  %0 = constant dense<[1, 0]> : tensor<2xi1>
  return
}
