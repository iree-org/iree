// RUN: iree-opt -iree-convert-to-hal %s --split-input-file | IreeFileCheck %s

// CHECK-LABEL: @dynamic_shape_constant
func @dynamic_shape_constant() {
  // CHECK: %dev = hal.ex.shared_device
  // CHECK: %allocator = hal.device.allocator %dev
  // CHECK: %view = hal.buffer_view.const %allocator, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch" : !hal.buffer_view = dense<2> : tensor<2xi32>
  // CHECK: %[[RES:.+]] = iree.do_not_optimize(%view) : !hal.buffer_view
  %c = iree.dynamic_shape_constant dense<2> : tensor<2xi32> -> tensor<?xi32>
  return
}
