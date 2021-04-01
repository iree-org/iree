// RUN: iree-opt -split-input-file -canonicalize -cse %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @allocator_constant_buffer
// CHECK-SAME: %[[ALLOCATOR:.+]]: !hal.allocator
func @allocator_constant_buffer(%allocator: !hal.allocator) -> !hal.buffer {
  //      CHECK: %[[RODATA:.+]] = iree.byte_buffer.constant : !iree.byte_buffer = dense<123> : tensor<4x4xi32>
  // CHECK-NEXT: %[[BUFFER:.+]] = hal.allocator.map<%[[ALLOCATOR]] : !hal.allocator>
  // CHECK-SAME:   source(%[[RODATA]] : !iree.byte_buffer)[%c0, %c-1]
  // CHECK-SAME:   type("HostVisible|DeviceVisible|DeviceLocal")
  // CHECK-SAME:   usage("Constant|Transfer|Mapping|Dispatch")
  // CHECK-SAME:   : !hal.buffer
  %ref = hal.allocator.constant<%allocator : !hal.allocator>
         type(DeviceLocal) usage(Transfer) : !hal.buffer =
         dense<123> : tensor<4x4xi32>
  // CHECK-NEXT: return %[[BUFFER]]
  return %ref : !hal.buffer
}

// -----

// CHECK-LABEL: @allocator_constant_buffer_view
// CHECK-SAME: %[[ALLOCATOR:.+]]: !hal.allocator
func @allocator_constant_buffer_view(%allocator: !hal.allocator) -> !hal.buffer_view {
  //      CHECK: %[[RODATA:.+]] = iree.byte_buffer.constant : !iree.byte_buffer = dense<123> : tensor<4x4xi32>
  // CHECK-NEXT: %[[BUFFER:.+]] = hal.allocator.map<%[[ALLOCATOR]] : !hal.allocator>
  // CHECK-SAME:   source(%[[RODATA]] : !iree.byte_buffer)[%c0, %c-1]
  // CHECK-SAME:   type("HostVisible|DeviceVisible|DeviceLocal")
  // CHECK-SAME:   usage("Constant|Transfer|Mapping|Dispatch")
  // CHECK-SAME:   : !hal.buffer
  // CHECK-NEXT: %[[VIEW:.+]] = hal.buffer_view.create %[[BUFFER]], element_type = %c16777248_i32, shape = [%c4, %c4] : !hal.buffer -> !hal.buffer_view
  %ref = hal.allocator.constant<%allocator : !hal.allocator>
         type(DeviceLocal) usage(Transfer) : !hal.buffer_view =
         dense<123> : tensor<4x4xi32>
  // CHECK-NEXT: return %[[VIEW]]
  return %ref : !hal.buffer_view
}
