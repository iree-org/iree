// RUN: iree-opt -iree-convert-to-hal %s --split-input-file | IreeFileCheck %s

// CHECK-LABEL: @dynamic_shape_constant
func @dynamic_shape_constant() {
  //      CHECK: %[[ALLOCATOR:.+]] = hal.device.allocator
  // CHECK-NEXT: %[[BUFFER:.+]] = hal.allocator.constant<%[[ALLOCATOR]] : !hal.allocator>
  // CHECK-SAME:   type("HostVisible|DeviceVisible|DeviceLocal")
  // CHECK-SAME:   usage("Constant|Transfer|Mapping|Dispatch")
  // CHECK-SAME:   : !hal.buffer = dense<2> : tensor<2xi32>
  //      CHECK: %[[VIEW:.+]] = hal.buffer_view.create
  // CHECK-SAME:   buffer(%[[BUFFER]] : !hal.buffer)
  // CHECK-SAME:   shape([%c2])
  // CHECK-SAME:   type(%c16777248_i32)
  // CHECK-SAME:   encoding(%c1_i32) : !hal.buffer_view
  // CHECK-NEXT: %[[RET:.+]] = util.do_not_optimize(%[[VIEW]]) : !hal.buffer_view
  %c = util.dynamic_shape_constant dense<2> : tensor<2xi32> -> tensor<?xi32>
  return
}
