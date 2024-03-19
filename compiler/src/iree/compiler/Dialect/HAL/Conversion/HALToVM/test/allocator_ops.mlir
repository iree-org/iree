// RUN: iree-opt --split-input-file --canonicalize --iree-convert-hal-to-vm %s | FileCheck %s

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
