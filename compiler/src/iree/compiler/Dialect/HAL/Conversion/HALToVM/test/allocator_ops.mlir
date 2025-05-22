// RUN: iree-opt --split-input-file --canonicalize --iree-vm-conversion %s | FileCheck %s

// CHECK-LABEL: vm.func private @allocatorSelect
// CHECK-SAME: (%[[DEVICE_A:.+]]: !vm.ref<!hal.device>, %[[AFFINITY_A:.+]]: i64, %[[DEVICE_B:.+]]: !vm.ref<!hal.device>, %[[AFFINITY_B:.+]]: i64)
util.func public @allocatorSelect(%device_a: !hal.device, %affinity_a: i64, %device_b: !hal.device, %affinity_b: i64) -> (!hal.device, i64) {
  // CHECK-DAG: %[[TYPE:.+]] = vm.const.i32 2
  %type = arith.constant 2 : i32
  // CHECK-DAG: %[[USAGE:.+]] = vm.const.i32 3
  %usage = arith.constant 3 : i32
  // CHECK-DAG: %[[FLAGS:.+]] = vm.const.i64.zero
  // CHECK: %[[DEVICE_AFFINITY:.+]]:2 = vm.call.variadic @hal.allocator.select(
  // CHECK-SAME: %[[TYPE]], %[[USAGE]], %[[FLAGS]],
  // CHECK-SAME: [(%[[DEVICE_A]], %[[AFFINITY_A]]), (%[[DEVICE_B]], %[[AFFINITY_B]])]
  %device, %queue_affinity = hal.allocator.select
      from([
        (%device_a, %affinity_a : !hal.device, i64),
        (%device_b, %affinity_b : !hal.device, i64)
      ])
      type(%type) usage(%usage) : !hal.device, i64
  // CHECK: vm.return %[[DEVICE_AFFINITY]]#0, %[[DEVICE_AFFINITY]]#1
  util.return %device, %queue_affinity : !hal.device, i64
}

// -----

// CHECK-LABEL: vm.func private @allocatorAllocate
util.func public @allocatorAllocate(%arg0 : !hal.allocator) -> !hal.buffer {
  // CHECK-DAG: %[[SIZE:.+]] = vm.const.i64 1024
  %size = arith.constant 1024 : index
  // CHECK-DAG: %[[AFFINITY:.+]] = vm.const.i64 -1
  %affinity = arith.constant -1 : i64
  // CHECK: %ref = vm.call @hal.allocator.allocate(%arg0, %[[AFFINITY]], %c70, %c3075, %[[SIZE]]) : (!vm.ref<!hal.allocator>, i64, i32, i32, i64) -> !vm.ref<!hal.buffer>
  %0 = hal.allocator.allocate<%arg0 : !hal.allocator> affinity(%affinity) type("HostLocal") usage("DispatchStorage|Transfer") : !hal.buffer{%size}
  util.return %0 : !hal.buffer
}

// -----

// CHECK-LABEL: vm.func private @allocatorImport
util.func public @allocatorImport(%arg0 : !hal.allocator, %arg1 : !util.buffer) -> (i1, !hal.buffer) {
  // CHECK-DAG: %[[OFFSET:.+]] = vm.const.i64 128
  %offset = arith.constant 128 : index
  // CHECK-DAG: %[[LENGTH:.+]] = vm.const.i64 256
  %length = arith.constant 256 : index
  // CHECK-DAG: %[[AFFINITY:.+]] = vm.const.i64 -1
  %affinity = arith.constant -1 : i64
  // CHECK: %[[IMPORTED:.+]] = vm.call @hal.allocator.import(%arg0, %c1, %[[AFFINITY]], %c6, %c3, %arg1, %[[OFFSET]], %[[LENGTH]]) : (!vm.ref<!hal.allocator>, i32, i64, i32, i32, !vm.buffer, i64, i64) -> !vm.ref<!hal.buffer>
  %did_import, %buffer = hal.allocator.import<%arg0 : !hal.allocator> source(%arg1 : !util.buffer)[%offset, %length] affinity(%affinity) type("HostVisible|HostCoherent") usage("Transfer") : i1, !hal.buffer
  // CHECK: %[[DID_IMPORT:.+]] = vm.cmp.nz.ref %[[IMPORTED]]
  // CHECK: vm.return %[[DID_IMPORT]], %[[IMPORTED]]
  util.return %did_import, %buffer : i1, !hal.buffer
}
