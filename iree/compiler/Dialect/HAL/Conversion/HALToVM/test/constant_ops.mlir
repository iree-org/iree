// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

// CHECK: vm.rodata private @pool_storage0 dense<[102, 102, 6, 64, -51, -52, 76, 64, -102, -103, -119, 64, -51, -52, -84, 64]> : vector<16xi8>
// CHECK: vm.rodata private @pool_storage1 dense<[6, 7, 8, 0]> : vector<4xi8>
hal.constant_pool @pool attributes {buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824, min_buffer_offset_alignment = 32, max_buffer_range = 134217728, min_buffer_range_alignment = 4>} {
  hal.constant_pool.span @cst0 : tensor<4xf32> = @_storage0[#hal.byte_range<0, 16>] -> @pool_storage0_buffer[#hal.byte_range<0, 16>]
  hal.constant_pool.span @cst1 : tensor<3xi8> = @_storage1[#hal.byte_range<0, 3>] -> @pool_storage1_buffer[#hal.byte_range<0, 3>]
  hal.constant_pool.splat @cst2 = dense<1.000000e+00> : tensor<1xf32> -> @pool_splats[#hal.byte_range<0, 4>]
  hal.constant_pool.splat @cst3 = dense<1234567890> : tensor<8xi32> -> @pool_splats[#hal.byte_range<32, 32>]
  hal.constant_storage @_storage0 = dense<[102, 102, 6, 64, -51, -52, 76, 64, -102, -103, -119, 64, -51, -52, -84, 64]> : vector<16xi8>
  hal.constant_storage @_storage1 = dense<[6, 7, 8, 0]> : vector<4xi8>
}

// CHECK: vm.global.ref @pool_storage0_buffer initializer(@pool_storage0_buffer_initializer) : !vm.ref<!hal.buffer>
hal.variable @pool_storage0_buffer init(@pool_storage0_buffer_initializer) : !hal.buffer attributes {sym_visibility = "private"}
// CHECK: vm.func private @pool_storage0_buffer_initializer() -> !vm.ref<!hal.buffer>
func private @pool_storage0_buffer_initializer() -> !hal.buffer {
  %c0 = constant 0 : index
  %c16 = constant 16 : index
  %dev = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator<%dev : !hal.device> : !hal.allocator
  // CHECK: [[STORAGE_REF:%.+]] = vm.const.ref.rodata @pool_storage0 : !vm.buffer
  %storage = hal.constant_storage.lookup @pool::@_storage0 : !iree.byte_buffer
  // CHECK: = vm.call @hal.allocator.wrap.byte_buffer({{.+}}, %c22, %c15, [[STORAGE_REF]], %zero, %c16)
  %mapped = hal.allocator.map<%allocator : !hal.allocator>
      source(%storage : !iree.byte_buffer)[%c0, %c16]
      type("HostVisible|HostCoherent|DeviceVisible")
      usage("Constant|Transfer|Mapping|Dispatch") : !hal.buffer
  return %mapped : !hal.buffer
}

// CHECK: vm.global.ref @pool_storage1_buffer initializer(@pool_storage1_buffer_initializer) : !vm.ref<!hal.buffer>
hal.variable @pool_storage1_buffer init(@pool_storage1_buffer_initializer) : !hal.buffer attributes {sym_visibility = "private"}
func private @pool_storage1_buffer_initializer() -> !hal.buffer

// CHECK: vm.global.ref @pool_splats initializer(@pool_splats_initializer) : !vm.ref<!hal.buffer>
hal.variable @pool_splats init(@pool_splats_initializer) : !hal.buffer attributes {sym_visibility = "private"}
// CHECK: vm.func private @pool_splats_initializer() -> !vm.ref<!hal.buffer>
func private @pool_splats_initializer() -> !hal.buffer {
  %c64 = constant 64 : index
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %c1065353216_i32 = constant 1065353216 : i32
  %c32 = constant 32 : index
  %c1234567890_i32 = constant 1234567890 : i32
  %dev = hal.ex.shared_device : !hal.device
  %allocator = hal.device.allocator<%dev : !hal.device> : !hal.allocator
  // CHECK: [[BUFFER:%.+]] = vm.call @hal.allocator.allocate({{.+}}, %c50, %c15, %c64)
  %buffer = hal.allocator.allocate<%allocator : !hal.allocator>
      type("HostVisible|DeviceVisible|DeviceLocal")
      usage("Constant|Transfer|Mapping|Dispatch") : !hal.buffer{%c64}
  return %buffer : !hal.buffer
}
