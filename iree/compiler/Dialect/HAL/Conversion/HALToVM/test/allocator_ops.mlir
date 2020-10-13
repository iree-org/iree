// RUN: iree-opt -split-input-file -canonicalize -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK-LABEL: @allocatorComputeSizeFoldsAway
func @allocatorComputeSizeFoldsAway(%arg0 : !hal.allocator) -> index {
  // CHECK: %c4194304 = vm.const.i32 4194304 : i32
  // CHECK-NOT: hal.allocator.compute_size
  %c1024 = constant 1024 : index
  %0 = hal.allocator.compute_size %arg0, shape=[%c1024, %c1024], element_type=32
  return %0 : index
}

// -----

// CHECK-LABEL: @allocatorAllocate
func @allocatorAllocate(%arg0 : !hal.allocator) -> !hal.buffer {
  %c1024 = constant 1024 : index
  // CHECK: %ref = vm.call @hal.allocator.allocate(%arg0, %c6, %c15, %c1024) : (!vm.ref<!hal.allocator>, i32, i32, i32) -> !vm.ref<!hal.buffer>
  %0 = hal.allocator.allocate %arg0, "HostLocal", "All", %c1024 : !hal.buffer
  return %0 : !hal.buffer
}

// -----

// CHECK-LABEL: func @allocatorMapByteBuffer
func @allocatorMapByteBuffer(%arg0 : !hal.allocator, %arg1 : !iree.byte_buffer) -> !hal.buffer {
  %offset = constant 128 : index
  %length = constant 256 : index
  // CHECK: = vm.call @hal.allocator.wrap.byte_buffer(%arg0, %c6, %c2, %arg1, %c128, %c256) : (!vm.ref<!hal.allocator>, i32, i32, !vm.ref<!iree.byte_buffer>, i32, i32) -> !vm.ref<!hal.buffer>
  %buffer = hal.allocator.map %arg0, "HostVisible|HostCoherent", "Transfer", %arg1[%offset, %length] : !iree.byte_buffer -> !hal.buffer
  return %buffer : !hal.buffer
}
