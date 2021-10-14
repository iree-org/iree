// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK: vm.rodata private @pool_storage0 dense<[102, 102, 6, 64, -51, -52, 76, 64, -102, -103, -119, 64, -51, -52, -84, 64]> : vector<16xi8>
// CHECK: vm.rodata private @pool_storage1 dense<[6, 7, 8, 0]> : vector<4xi8>
hal.constant_pool @pool attributes {buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824, min_buffer_offset_alignment = 32, max_buffer_range = 134217728, min_buffer_range_alignment = 4>} {
  hal.constant_pool.span @cst0 : tensor<4xf32> = @_storage0[#util.byte_range<0, 16>] -> @pool_storage0_buffer[#util.byte_range<0, 16>]
  hal.constant_pool.span @cst1 : tensor<3xi8> = @_storage1[#util.byte_range<0, 3>] -> @pool_storage1_buffer[#util.byte_range<0, 3>]
  hal.constant_pool.splat @cst2 = dense<1.000000e+00> : tensor<1xf32> -> @pool_splats[#util.byte_range<0, 4>]
  hal.constant_pool.splat @cst3 = dense<1234567890> : tensor<8xi32> -> @pool_splats[#util.byte_range<32, 32>]
  hal.constant_storage @_storage0 = dense<[102, 102, 6, 64, -51, -52, 76, 64, -102, -103, -119, 64, -51, -52, -84, 64]> : vector<16xi8>
  hal.constant_storage @_storage1 = dense<[6, 7, 8, 0]> : vector<4xi8>
}

// CHECK: vm.global.ref private @pool_storage0_buffer : !vm.ref<!hal.buffer>
util.global private @pool_storage0_buffer : !hal.buffer
// CHECK-NEXT: vm.initializer {
util.initializer {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %dev = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator<%dev : !hal.device> : !hal.allocator
  // CHECK: %[[STORAGE_REF:.+]] = vm.const.ref.rodata @pool_storage0 : !vm.buffer
  %storage = hal.constant_storage.lookup @pool::@_storage0 : !util.byte_buffer
  // CHECK: %[[STORAGE_BUFFER:.+]] = vm.call @hal.allocator.wrap.byte_buffer({{.+}}, %c22, %c15, %[[STORAGE_REF]], %zero, %c16)
  %mapped = hal.allocator.map<%allocator : !hal.allocator>
      source(%storage : !util.byte_buffer)[%c0, %c16]
      type("HostVisible|HostCoherent|DeviceVisible")
      usage("Constant|Transfer|Mapping|Dispatch") : !hal.buffer
  // CHECK-NEXT:   vm.global.store.ref %[[STORAGE_BUFFER]], @pool_storage0_buffer : !vm.ref<!hal.buffer>
  util.global.store %mapped, @pool_storage0_buffer : !hal.buffer
  util.initializer.return
}

// CHECK: vm.global.ref private @pool_storage1_buffer : !vm.ref<!hal.buffer>
util.global private @pool_storage1_buffer : !hal.buffer

// CHECK: vm.global.ref private @pool_splats : !vm.ref<!hal.buffer>
util.global private @pool_splats : !hal.buffer
// CHECK: vm.initializer {
util.initializer {
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1065353216_i32 = arith.constant 1065353216 : i32
  %c32 = arith.constant 32 : index
  %c1234567890_i32 = arith.constant 1234567890 : i32
  %dev = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator<%dev : !hal.device> : !hal.allocator
  // CHECK: [[BUFFER:%.+]] = vm.call @hal.allocator.allocate({{.+}}, %c50, %c15, %c64)
  %buffer = hal.allocator.allocate<%allocator : !hal.allocator>
      type("HostVisible|DeviceVisible|DeviceLocal")
      usage("Constant|Transfer|Mapping|Dispatch") : !hal.buffer{%c64}
  util.global.store %buffer, @pool_splats : !hal.buffer
  util.initializer.return
}
