// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @allocator_allocate
//  CHECK-SAME: (%[[ALLOCATOR:.+]]: !hal.allocator)
util.func public @allocator_allocate(%allocator: !hal.allocator) {
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  %affinity = arith.constant -1 : i64
  // CHECK-DAG: %[[SIZE:.+]] = arith.constant 123
  %size = arith.constant 123 : index
  //      CHECK: %[[REF:.+]] = hal.allocator.allocate<%[[ALLOCATOR]] : !hal.allocator>
  // CHECK-SAME:   type("HostVisible|HostCoherent|HostLocal")
  // CHECK-SAME:   usage("TransferSource|TransferTarget|Transfer")
  // CHECK-SAME:   : !hal.buffer{%[[SIZE]]}
  %ref = hal.allocator.allocate<%allocator : !hal.allocator>
      affinity(%affinity) type(HostLocal) usage(Transfer) : !hal.buffer{%size}
  util.return
}

// -----

// CHECK-LABEL: @allocator_import
//  CHECK-SAME: %[[ALLOCATOR:.+]]: !hal.allocator
util.func public @allocator_import(%allocator: !hal.allocator, %arg1: !util.buffer) {
  // CHECK-DAG: %[[OFFSET:.+]] = arith.constant 100
  %offset = arith.constant 100 : index
  // CHECK-DAG: %[[LENGTH:.+]] = arith.constant 200
  %length = arith.constant 200 : index
  // CHECK-DAG: %[[AFFINITY:.+]] = arith.constant -1
  %affinity = arith.constant -1 : i64
  //      CHECK: = hal.allocator.import<%[[ALLOCATOR]] : !hal.allocator>
  // CHECK-SAME:   source(%arg1 : !util.buffer)[%[[OFFSET]], %[[LENGTH]]]
  // CHECK-SAME:   affinity(%[[AFFINITY]])
  // CHECK-SAME:   type("DeviceVisible|DeviceLocal")
  // CHECK-SAME:   usage("TransferSource|TransferTarget|Transfer")
  // CHECK-SAME:   : i1, !hal.buffer
  %ok, %ref = hal.allocator.import<%allocator : !hal.allocator>
      source(%arg1 : !util.buffer)[%offset, %length]
      affinity(%affinity) type(DeviceLocal) usage(Transfer) : i1, !hal.buffer
  util.return
}
