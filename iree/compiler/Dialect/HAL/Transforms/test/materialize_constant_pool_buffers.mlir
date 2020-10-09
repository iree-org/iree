// RUN: iree-opt -split-input-file -iree-hal-materialize-constant-pool-buffers %s | IreeFileCheck %s

// CHECK-LABEL: hal.constant_pool @dense_variable_init
hal.constant_pool @dense_variable_init attributes {buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824, min_buffer_offset_alignment = 32, max_buffer_range = 134217728, min_buffer_range_alignment = 4>} {
  // CHECK-NEXT: @cst0 {{.+}} -> @dense_variable_init_storage_buffer[#hal.byte_range<0, 512>]
  hal.constant_pool.span @cst0 : tensor<128xf32> = @_storage[#hal.byte_range<0, 512>]
  // CHECK-NEXT: @cst1 {{.+}} -> @dense_variable_init_storage_buffer[#hal.byte_range<512, 256>]
  hal.constant_pool.span @cst1 : tensor<64xf32> = @_storage[#hal.byte_range<512, 256>]
  hal.constant_storage @_storage = dense<1> : vector<768xi8>
}

//      CHECK: hal.variable @dense_variable_init_storage_buffer init(@dense_variable_init_storage_buffer_initializer) : !hal.buffer
// CHECK-NEXT: func @dense_variable_init_storage_buffer_initializer() -> !hal.buffer
//      CHECK: [[STORAGE:%.+]] = hal.constant_storage.lookup @dense_variable_init::@_storage : !iree.byte_buffer
//      CHECK: = hal.allocator.map {{.+}} [[STORAGE]][%c0, %c768] : !iree.byte_buffer -> !hal.buffer

// -----

// CHECK-LABEL: hal.constant_pool @splat_variable_init
hal.constant_pool @splat_variable_init attributes {buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824, min_buffer_offset_alignment = 32, max_buffer_range = 134217728, min_buffer_range_alignment = 4>} {
  // CHECK-NEXT: @cst0 {{.+}} -> @splat_variable_init_splats[#hal.byte_range<0, 4>]
  hal.constant_pool.splat @cst0 = dense<1.000000e+00> : tensor<1xf32>
  // CHECK-NEXT: @cst1 {{.+}} -> @splat_variable_init_splats[#hal.byte_range<32, 32>]
  hal.constant_pool.splat @cst1 = dense<1234567890> : tensor<8xi32>
}

//      CHECK: hal.variable @splat_variable_init_splats init(@splat_variable_init_splats_initializer) : !hal.buffer
// CHECK-NEXT: func @splat_variable_init_splats_initializer() -> !hal.buffer
//      CHECK: [[BUFFER:%.+]] = hal.allocator.allocate {{.+}} %c64 : !hal.buffer
//      CHECK: hal.buffer.fill [[BUFFER]], %c0, %c4, %c1065353216_i32
//      CHECK: hal.buffer.fill [[BUFFER]], %c32, %c32_0, %c1234567890_i32

// -----

// CHECK-LABEL: hal.constant_pool @pool
hal.constant_pool @pool attributes {buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824, min_buffer_offset_alignment = 32, max_buffer_range = 134217728, min_buffer_range_alignment = 4>} {
  // CHECK-NEXT: @cst0 {{.+}} -> @pool_storage0_buffer[#hal.byte_range<0, 16>]
  hal.constant_pool.span @cst0 : tensor<4xf32> = @_storage0[#hal.byte_range<0, 16>]
  // CHECK-NEXT: @cst1 {{.+}} -> @pool_storage1_buffer[#hal.byte_range<0, 3>]
  hal.constant_pool.span @cst1 : tensor<3xi8> = @_storage1[#hal.byte_range<0, 3>]
  // CHECK-NEXT: @cst2 {{.+}} -> @pool_splats[#hal.byte_range<0, 4>]
  hal.constant_pool.splat @cst2 = dense<1.000000e+00> : tensor<1xf32>
  // CHECK-NEXT: @cst3 {{.+}} -> @pool_splats[#hal.byte_range<32, 32>]
  hal.constant_pool.splat @cst3 = dense<1234567890> : tensor<8xi32>
  hal.constant_storage @_storage0 = dense<[102, 102, 6, 64, -51, -52, 76, 64, -102, -103, -119, 64, -51, -52, -84, 64]> : vector<16xi8>
  hal.constant_storage @_storage1 = dense<[6, 7, 8, 0]> : vector<4xi8>
}

//      CHECK: hal.variable @pool_storage0_buffer init(@pool_storage0_buffer_initializer) : !hal.buffer
// CHECK-NEXT: func @pool_storage0_buffer_initializer() -> !hal.buffer
//      CHECK: [[STORAGE:%.+]] = hal.constant_storage.lookup @pool::@_storage0 : !iree.byte_buffer
//      CHECK: = hal.allocator.map {{.+}} [[STORAGE]][%c0, %c16] : !iree.byte_buffer -> !hal.buffer

//      CHECK: hal.variable @pool_storage1_buffer init(@pool_storage1_buffer_initializer) : !hal.buffer
// CHECK-NEXT: func @pool_storage1_buffer_initializer() -> !hal.buffer

//      CHECK: hal.variable @pool_splats init(@pool_splats_initializer) : !hal.buffer
// CHECK-NEXT: func @pool_splats_initializer() -> !hal.buffer
//      CHECK: [[BUFFER:%.+]] = hal.allocator.allocate %allocator, "HostVisible|DeviceVisible|DeviceLocal", "Constant|Transfer|Mapping|Dispatch", %c64 : !hal.buffer
//      CHECK: hal.buffer.fill [[BUFFER]], %c0, %c4, %c1065353216_i32
//      CHECK: hal.buffer.fill [[BUFFER]], %c32, %c32_0, %c1234567890_i32
