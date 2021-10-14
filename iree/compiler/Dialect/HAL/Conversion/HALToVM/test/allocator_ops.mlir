// RUN: iree-opt -split-input-file -canonicalize -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK-LABEL: vm.func private @allocatorComputeSizeFoldsAway
func @allocatorComputeSizeFoldsAway(%arg0 : !hal.allocator) -> index {
  // CHECK: %c4194304 = vm.const.i32 4194304 : i32
  // CHECK-NOT: hal.allocator.compute_size
  %c1024 = arith.constant 1024 : index
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %0 = hal.allocator.compute_size<%arg0 : !hal.allocator>
      shape([%c1024, %c1024]) type(%c32_i32) encoding(%c1_i32) : index
  return %0 : index
}

// -----

// CHECK-LABEL: vm.func private @allocatorAllocate
func @allocatorAllocate(%arg0 : !hal.allocator) -> !hal.buffer {
  %c1024 = arith.constant 1024 : index
  // CHECK: %ref = vm.call @hal.allocator.allocate(%arg0, %c6, %c14, %c1024) : (!vm.ref<!hal.allocator>, i32, i32, i32) -> !vm.ref<!hal.buffer>
  %0 = hal.allocator.allocate<%arg0 : !hal.allocator> type("HostLocal") usage("All") : !hal.buffer{%c1024}
  return %0 : !hal.buffer
}

// -----

// CHECK-LABEL: vm.func private @allocatorMapByteBuffer
func @allocatorMapByteBuffer(%arg0 : !hal.allocator, %arg1 : !util.byte_buffer) -> !hal.buffer {
  %offset = arith.constant 128 : index
  %length = arith.constant 256 : index
  // CHECK: = vm.call @hal.allocator.wrap.byte_buffer(%arg0, %c6, %c2, %arg1, %c128, %c256) : (!vm.ref<!hal.allocator>, i32, i32, !vm.buffer, i32, i32) -> !vm.ref<!hal.buffer>
  %buffer = hal.allocator.map<%arg0 : !hal.allocator> source(%arg1 : !util.byte_buffer)[%offset, %length] type("HostVisible|HostCoherent") usage(Transfer) : !hal.buffer
  return %buffer : !hal.buffer
}
