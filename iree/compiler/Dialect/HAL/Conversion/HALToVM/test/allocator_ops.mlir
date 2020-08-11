// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK-LABEL: @allocatorComputeSize
func @allocatorComputeSize(%arg0 : !hal.allocator) -> index {
  %c1024 = constant 1024 : index
  // CHECK: %0 = vm.call.variadic @hal.allocator.compute_size(%arg0, [%c1024, %c1024], %c32) : (!vm.ref<!hal.allocator>, i32 ..., i32) -> i32
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

// CHECK: vm.rodata @allocatorAllocateConst_const_0 dense<123> : tensor<4x4xi32>
// CHECK-LABEL: func @allocatorAllocateConst
func @allocatorAllocateConst(%arg0 : !hal.allocator) -> !hal.buffer {
  // CHECK: %allocatorAllocateConst_const_0 = vm.const.ref.rodata @allocatorAllocateConst_const_0 : !vm.ref<!iree.byte_buffer>
  // CHECK: %ref = vm.call.variadic @hal.allocator.allocate.const(%arg0, %c6, %c2, [%c4, %c4_0], %c16777248, %allocatorAllocateConst_const_0) : (!vm.ref<!hal.allocator>, i32, i32, i32 ..., i32, !vm.ref<!iree.byte_buffer>) -> !vm.ref<!hal.buffer>
  %buffer = hal.allocator.allocate.const %arg0, "HostVisible|HostCoherent", "Transfer" : !hal.buffer = dense<123> : tensor<4x4xi32>
  return %buffer : !hal.buffer
}
