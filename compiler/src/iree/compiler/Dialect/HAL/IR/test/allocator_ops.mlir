// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @allocator_select_attr
util.func public @allocator_select_attr() -> (!hal.device, i64) {
  // CHECK: %[[DEVICE:.+]], %[[QUEUE_AFFINITY:.+]] = hal.allocator.select.attr
  // CHECK-SAME:   from(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
  // CHECK-SAME:   type("HostVisible|HostCoherent|HostLocal")
  // CHECK-SAME:   usage("TransferSource|TransferTarget|Transfer")
  // CHECK-SAME:   : !hal.device, i64
  %device, %queue_affinity = hal.allocator.select.attr
      from(#hal.device.optimal<[#hal.device.affinity<@device_a>, #hal.device.affinity<@device_b>]>)
      type(HostLocal) usage(Transfer) : !hal.device, i64
  util.return %device, %queue_affinity : !hal.device, i64
}

// -----

// CHECK-LABEL: @allocator_select
// CHECK-SAME: (%[[DEVICE_A:.+]]: !hal.device, %[[AFFINITY_A:.+]]: i64, %[[DEVICE_B:.+]]: !hal.device, %[[AFFINITY_B:.+]]: i64)
util.func public @allocator_select(%device_a: !hal.device, %affinity_a: i64, %device_b: !hal.device, %affinity_b: i64) -> (!hal.device, i64) {
  // CHECK-DAG: %[[TYPE:.+]] = arith.constant 2
  %type = arith.constant 2 : i32
  // CHECK-DAG: %[[USAGE:.+]] = arith.constant 3
  %usage = arith.constant 3 : i32
  // CHECK: %[[DEVICE:.+]], %[[QUEUE_AFFINITY:.+]] = hal.allocator.select
  // CHECK-SAME:   from([
  // CHECK-NEXT:     (%[[DEVICE_A]], %[[AFFINITY_A]] : !hal.device, i64),
  // CHECK-NEXT:     (%[[DEVICE_B]], %[[AFFINITY_B]] : !hal.device, i64)
  // CHECK-NEXT:   ])
  // CHECK-SAME:   type(%[[TYPE]])
  // CHECK-SAME:   usage(%[[USAGE]])
  // CHECK-SAME:   : !hal.device, i64
  %device, %queue_affinity = hal.allocator.select
      from([
        (%device_a, %affinity_a : !hal.device, i64),
        (%device_b, %affinity_b : !hal.device, i64)
      ])
      type(%type) usage(%usage) : !hal.device, i64
  util.return %device, %queue_affinity : !hal.device, i64
}

// -----

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
