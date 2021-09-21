// RUN: iree-opt -split-input-file -iree-hal-materialize-constant-pool-buffers %s | IreeFileCheck %s

// CHECK-LABEL: hal.constant_pool public @dense_variable_init
hal.constant_pool @dense_variable_init attributes {buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824, min_buffer_offset_alignment = 32, max_buffer_range = 134217728, min_buffer_range_alignment = 4>} {
  // CHECK-NEXT: @cst0 {{.+}} -> @dense_variable_init_storage_buffer[#util.byte_range<0, 512>]
  hal.constant_pool.span @cst0 : tensor<128xf32> = @_storage[#util.byte_range<0, 512>]
  // CHECK-NEXT: @cst1 {{.+}} -> @dense_variable_init_storage_buffer[#util.byte_range<512, 256>]
  hal.constant_pool.span @cst1 : tensor<64xf32> = @_storage[#util.byte_range<512, 256>]
  hal.constant_storage @_storage = dense<1> : vector<768xi8>
}

//      CHECK: util.global private @dense_variable_init_storage_buffer : !hal.buffer
// CHECK-NEXT: util.initializer {
//      CHECK: %[[STORAGE:.+]] = hal.constant_storage.lookup @dense_variable_init::@_storage : !util.byte_buffer
//      CHECK: = hal.allocator.map<%allocator : !hal.allocator>
// CHECK-SAME:   source(%[[STORAGE]] : !util.byte_buffer)[%c0, %c768]
// CHECK-SAME:   : !hal.buffer

// -----

// CHECK-LABEL: hal.constant_pool public @splat_variable_init
hal.constant_pool @splat_variable_init attributes {buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824, min_buffer_offset_alignment = 32, max_buffer_range = 134217728, min_buffer_range_alignment = 4>} {
  // CHECK-NEXT: @cst0 {{.+}} -> @splat_variable_init_splats[#util.byte_range<0, 4>]
  hal.constant_pool.splat @cst0 = dense<1.000000e+00> : tensor<1xf32>
  // CHECK-NEXT: @cst1 {{.+}} -> @splat_variable_init_splats[#util.byte_range<32, 32>]
  hal.constant_pool.splat @cst1 = dense<1234567890> : tensor<8xi32>
}

//      CHECK: util.global private @splat_variable_init_splats : !hal.buffer
// CHECK-NEXT: util.initializer {
//      CHECK: %[[BUFFER:.+]] = hal.allocator.allocate<%allocator : !hal.allocator>
// CHECK-SAME:   type("HostVisible|DeviceVisible|DeviceLocal")
// CHECK-SAME:   usage("Constant|Transfer|Mapping|Dispatch") : !hal.buffer{%c64}
//      CHECK: hal.command_buffer.fill_buffer<%cmd : !hal.command_buffer>
// CHECK-SAME:   target(%[[BUFFER]] : !hal.buffer)[%c0, %c4]
// CHECK-SAME:   pattern(%c1065353216_i32 : i32)
//      CHECK: hal.command_buffer.fill_buffer<%cmd : !hal.command_buffer>
// CHECK-SAME:   target(%[[BUFFER]] : !hal.buffer)[%c32, %c32_0]
// CHECK-SAME:   pattern(%c1234567890_i32 : i32)

// -----

// CHECK-LABEL: hal.constant_pool public @pool
hal.constant_pool @pool attributes {buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824, min_buffer_offset_alignment = 32, max_buffer_range = 134217728, min_buffer_range_alignment = 4>} {
  // CHECK-NEXT: @cst0 {{.+}} -> @pool_storage0_buffer[#util.byte_range<0, 16>]
  hal.constant_pool.span @cst0 : tensor<4xf32> = @_storage0[#util.byte_range<0, 16>]
  // CHECK-NEXT: @cst1 {{.+}} -> @pool_storage1_buffer[#util.byte_range<0, 3>]
  hal.constant_pool.span @cst1 : tensor<3xi8> = @_storage1[#util.byte_range<0, 3>]
  // CHECK-NEXT: @cst2 {{.+}} -> @pool_splats[#util.byte_range<0, 4>]
  hal.constant_pool.splat @cst2 = dense<1.000000e+00> : tensor<1xf32>
  // CHECK-NEXT: @cst3 {{.+}} -> @pool_splats[#util.byte_range<32, 32>]
  hal.constant_pool.splat @cst3 = dense<1234567890> : tensor<8xi32>
  hal.constant_storage @_storage0 = dense<[102, 102, 6, 64, -51, -52, 76, 64, -102, -103, -119, 64, -51, -52, -84, 64]> : vector<16xi8>
  hal.constant_storage @_storage1 = dense<[6, 7, 8, 0]> : vector<4xi8>
}

//      CHECK: util.global private @pool_storage0_buffer : !hal.buffer
// CHECK-NEXT: util.initializer {
//      CHECK: %[[STORAGE:.+]] = hal.constant_storage.lookup @pool::@_storage0 : !util.byte_buffer
//      CHECK: = hal.allocator.map<%allocator : !hal.allocator>
// CHECK-SAME:   source(%[[STORAGE]] : !util.byte_buffer)[%c0, %c16]
// CHECK-SAME:   : !hal.buffer

//      CHECK: util.global private @pool_storage1_buffer : !hal.buffer
// CHECK-NEXT: util.initializer {

//      CHECK: util.global private @pool_splats : !hal.buffer
// CHECK-NEXT: util.initializer {
//      CHECK: %[[BUFFER:.+]] = hal.allocator.allocate<%allocator : !hal.allocator>
// CHECK-SAME:   : !hal.buffer{%c64}
//      CHECK: hal.command_buffer.fill_buffer<%cmd : !hal.command_buffer>
// CHECK-SAME:   target(%[[BUFFER]] : !hal.buffer)[%c0, %c4]
// CHECK-SAME:   pattern(%c1065353216_i32 : i32)
//      CHECK: hal.command_buffer.fill_buffer<%cmd : !hal.command_buffer>
// CHECK-SAME:   target(%[[BUFFER]] : !hal.buffer)[%c32, %c32_0]
// CHECK-SAME:   pattern(%c1234567890_i32 : i32)
