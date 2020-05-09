// RUN: iree-opt --iree-convert-flow-to-hal %s --split-input-file | IreeFileCheck %s

// CHECK-LABEL: @preserve_compiler_hints
func @preserve_compiler_hints() {
  // CHECK: iree.do_not_optimize()
  iree.do_not_optimize()

  // CHECK: %[[C:.+]] = constant 2
  %c = constant 2 : i32
  // CHECK: iree.do_not_optimize(%[[C]])
  iree.do_not_optimize(%c) : i32
  return
}

// -----

// CHECK-LABEL: @dynamic_shape_constant
func @dynamic_shape_constant() {
  // CHECK: %dev = hal.ex.shared_device
  // CHECK: %allocator = hal.device.allocator %dev
  // CHECK: %view = hal.buffer_view.const %allocator, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch" : !hal.buffer_view = dense<2> : tensor<2xi32>
  // CHECK: %[[RES:.+]] = iree.do_not_optimize(%view) : !hal.buffer_view
  %c = iree.dynamic_shape_constant dense<2> : tensor<2xi32> -> tensor<?xi32>
  return
}
