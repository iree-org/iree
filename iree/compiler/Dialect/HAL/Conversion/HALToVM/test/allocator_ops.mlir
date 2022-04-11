// RUN: iree-opt -split-input-file -canonicalize -iree-convert-hal-to-vm %s | FileCheck %s

// CHECK-LABEL: vm.func private @allocatorAllocate
func.func @allocatorAllocate(%arg0 : !hal.allocator) -> !hal.buffer {
  %c1024 = arith.constant 1024 : index
  // CHECK: %ref = vm.call @hal.allocator.allocate(%arg0, %c6, %c10, %c1024) : (!vm.ref<!hal.allocator>, i32, i32, i32) -> !vm.ref<!hal.buffer>
  %0 = hal.allocator.allocate<%arg0 : !hal.allocator> type("HostLocal") usage("Dispatch|Transfer") : !hal.buffer{%c1024}
  return %0 : !hal.buffer
}

// -----

// CHECK-LABEL: vm.func private @allocatorMapByteBuffer
func.func @allocatorMapByteBuffer(%arg0 : !hal.allocator, %arg1 : !util.byte_buffer) -> !hal.buffer {
  %offset = arith.constant 128 : index
  %length = arith.constant 256 : index
  // CHECK: = vm.call @hal.allocator.wrap.byte_buffer(%arg0, %c6, %c2, %arg1, %c128, %c256) : (!vm.ref<!hal.allocator>, i32, i32, !vm.buffer, i32, i32) -> !vm.ref<!hal.buffer>
  %buffer = hal.allocator.map<%arg0 : !hal.allocator> source(%arg1 : !util.byte_buffer)[%offset, %length] type("HostVisible|HostCoherent") usage(Transfer) : !hal.buffer
  return %buffer : !hal.buffer
}
