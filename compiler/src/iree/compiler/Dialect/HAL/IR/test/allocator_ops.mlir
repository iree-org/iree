// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @allocator_allocate
//  CHECK-SAME: (%[[ALLOCATOR:.+]]: !hal.allocator)
func.func @allocator_allocate(%allocator: !hal.allocator) {
  // CHECK-DAG: %[[SIZE:.+]] = arith.constant 123
  %size = arith.constant 123 : index
  //      CHECK: %[[REF:.+]] = hal.allocator.allocate<%[[ALLOCATOR]] : !hal.allocator>
  // CHECK-SAME:   type("HostVisible|HostCoherent|HostLocal")
  // CHECK-SAME:   usage("TransferSource|TransferTarget|Transfer")
  // CHECK-SAME:   : !hal.buffer{%[[SIZE]]}
  %ref = hal.allocator.allocate<%allocator : !hal.allocator>
      type(HostLocal) usage(Transfer) : !hal.buffer{%size}
  return
}

// -----

// CHECK-LABEL: @allocator_map_byte_buffer
//  CHECK-SAME: %[[ALLOCATOR:.+]]: !hal.allocator
func.func @allocator_map_byte_buffer(%allocator: !hal.allocator, %arg1: !util.byte_buffer) {
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 100
  %offset = arith.constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 200
  %length = arith.constant 200 : index
  //      CHECK: = hal.allocator.map<%[[ALLOCATOR]] : !hal.allocator>
  // CHECK-SAME:   source(%arg1 : !util.byte_buffer)[%[[OFFSET]], %[[LENGTH]]]
  // CHECK-SAME:   type("DeviceVisible|DeviceLocal")
  // CHECK-SAME:   usage("TransferSource|TransferTarget|Transfer")
  // CHECK-SAME:   : !hal.buffer
  %ref = hal.allocator.map<%allocator : !hal.allocator>
                    source(%arg1 : !util.byte_buffer)[%offset, %length]
                    type(DeviceLocal) usage(Transfer) : !hal.buffer
  return
}
