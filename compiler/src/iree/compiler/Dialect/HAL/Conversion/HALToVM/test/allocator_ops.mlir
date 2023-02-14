// RUN: iree-opt --split-input-file --canonicalize --iree-convert-hal-to-vm %s | FileCheck %s

// CHECK-LABEL: vm.func private @allocatorAllocate
func.func @allocatorAllocate(%arg0 : !hal.allocator) -> !hal.buffer {
  // CHECK: %[[SIZE:.+]] = vm.const.i64 1024
  %c1024 = arith.constant 1024 : index
  // CHECK: %ref = vm.call @hal.allocator.allocate(%arg0, %c70, %c3075, %[[SIZE]]) : (!vm.ref<!hal.allocator>, i32, i32, i64) -> !vm.ref<!hal.buffer>
  %0 = hal.allocator.allocate<%arg0 : !hal.allocator> type("HostLocal") usage("DispatchStorage|Transfer") : !hal.buffer{%c1024}
  return %0 : !hal.buffer
}

// -----

// CHECK-LABEL: vm.func private @allocatorMapByteBuffer
func.func @allocatorMapByteBuffer(%arg0 : !hal.allocator, %arg1 : !util.buffer) -> !hal.buffer {
  // CHECK-DAG: %[[OFFSET:.+]] = vm.const.i64 128
  %offset = arith.constant 128 : index
  // CHECK-DAG: %[[LENGTH:.+]] = vm.const.i64 256
  %length = arith.constant 256 : index
  // CHECK: = vm.call @hal.allocator.allocate.initialized(%arg0, %c6, %c3, %arg1, %[[OFFSET]], %[[LENGTH]]) : (!vm.ref<!hal.allocator>, i32, i32, !vm.buffer, i64, i64) -> !vm.ref<!hal.buffer>
  %buffer = hal.allocator.allocate.initialized<%arg0 : !hal.allocator> source(%arg1 : !util.buffer)[%offset, %length] type("HostVisible|HostCoherent") usage("Transfer") : !hal.buffer
  return %buffer : !hal.buffer
}
