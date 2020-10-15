// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: hal.constant_pool @pool0
hal.constant_pool @pool0 attributes {
    buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824,
                                                 min_buffer_offset_alignment = 32,
                                                 max_buffer_range = 134217728,
                                                 min_buffer_range_alignment = 4>
  } {
  // CHECK-NEXT: hal.constant_pool.value @cst0 = dense<0.{{.+}}> : tensor<2x3xf32>
  hal.constant_pool.value @cst0 = dense<0.0> : tensor<2x3xf32>
  // CHECK-NEXT: hal.constant_pool.value @cst1 = dense<1.{{.+}}> : tensor<3x2xf32>
  hal.constant_pool.value @cst1 = dense<1.0> : tensor<3x2xf32>
}

// CHECK-LABEL: func @pools_identified()
func @pools_identified() -> (tensor<2x3xf32>, tensor<3x2xf32>) {
  // CHECK-NEXT: = hal.constant_pool.load @pool0::@cst0 : tensor<2x3xf32>
  %cst0 = hal.constant_pool.load @pool0::@cst0 : tensor<2x3xf32>
  // CHECK-NEXT: = hal.constant_pool.load @pool0::@cst1 : tensor<3x2xf32>
  %cst1 = hal.constant_pool.load @pool0::@cst1 : tensor<3x2xf32>
  return %cst0, %cst1 : tensor<2x3xf32>, tensor<3x2xf32>
}

// -----

// CHECK-LABEL: hal.constant_pool @storage_allocated
hal.constant_pool @storage_allocated attributes {
    buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824,
                                                 min_buffer_offset_alignment = 32,
                                                 max_buffer_range = 134217728,
                                                 min_buffer_range_alignment = 4>
  } {
  // CHECK-NEXT: hal.constant_pool.span @cst0 : tensor<2x3xf32> = @_storage[#hal.byte_range<0, 1024>]
  hal.constant_pool.span @cst0 : tensor<2x3xf32> = @_storage[#hal.byte_range<0, 1024>]
  // CHECK-NEXT: hal.constant_pool.span @cst1 : tensor<3x2xf32> = @_storage[#hal.byte_range<1024, 1024>]
  hal.constant_pool.span @cst1 : tensor<3x2xf32> = @_storage[#hal.byte_range<1024, 1024>]
  // CHECK-NEXT: hal.constant_pool.splat @cst2 = dense<1.000000e+00> : tensor<1xf32>
  hal.constant_pool.splat @cst2 = dense<1.000000e+00> : tensor<1xf32>
  // CHECK-NEXT: hal.constant_storage @_storage = dense<1> : vector<768xi8>
  hal.constant_storage @_storage = dense<1> : vector<768xi8>
}

// -----

// CHECK-LABEL: hal.constant_pool @pool
// CHECK-SAME: buffer_constraints = #hal.buffer_constraints
hal.constant_pool @pool attributes {
    buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824,
                                                 min_buffer_offset_alignment = 32,
                                                 max_buffer_range = 134217728,
                                                 min_buffer_range_alignment = 4>
  } {
  // CHECK-NEXT: hal.constant_pool.span @cst0 : tensor<4xf32> = @_storage0[#hal.byte_range<0, 16>] -> @pool_storage0_buffer[#hal.byte_range<0, 16>]
  hal.constant_pool.span @cst0 : tensor<4xf32> = @_storage0[#hal.byte_range<0, 16>] -> @pool_storage0_buffer[#hal.byte_range<0, 16>]
  // CHECK-NEXT: hal.constant_pool.span @cst1 : tensor<3xi8> = @_storage1[#hal.byte_range<0, 3>] -> @pool_storage1_buffer[#hal.byte_range<0, 3>]
  hal.constant_pool.span @cst1 : tensor<3xi8> = @_storage1[#hal.byte_range<0, 3>] -> @pool_storage1_buffer[#hal.byte_range<0, 3>]
  // CHECK-NEXT: hal.constant_pool.splat @cst2 = dense<1.000000e+00> : tensor<1xf32> -> @pool_splats[#hal.byte_range<0, 4>]
  hal.constant_pool.splat @cst2 = dense<1.000000e+00> : tensor<1xf32> -> @pool_splats[#hal.byte_range<0, 4>]
  // CHECK-NEXT: hal.constant_pool.splat @cst3 = dense<1234567890> : tensor<8xi32> -> @pool_splats[#hal.byte_range<32, 32>]
  hal.constant_pool.splat @cst3 = dense<1234567890> : tensor<8xi32> -> @pool_splats[#hal.byte_range<32, 32>]
  // CHECK-NEXT: hal.constant_storage @_storage0 = dense<[102, 102, 6, 64, -51, -52, 76, 64, -102, -103, -119, 64, -51, -52, -84, 64]> : vector<16xi8>
  hal.constant_storage @_storage0 = dense<[102, 102, 6, 64, -51, -52, 76, 64, -102, -103, -119, 64, -51, -52, -84, 64]> : vector<16xi8>
  // CHECK-NEXT: hal.constant_storage @_storage1 = dense<[6, 7, 8, 0]> : vector<4xi8>
  hal.constant_storage @_storage1 = dense<[6, 7, 8, 0]> : vector<4xi8>
}

// CHECK: func @storage_lookup
func @storage_lookup() {
  // CHECK-NEXT: = hal.constant_storage.lookup @pool::@_storage1 : !iree.byte_buffer
  %storage = hal.constant_storage.lookup @pool::@_storage1 : !iree.byte_buffer
  return
}

// -----

// CHECK: hal.variable @storage0_buffer0 : !hal.buffer
hal.variable @storage0_buffer0 : !hal.buffer
// CHECK-LABEL: func @runtime_buffer_subspan()
func @runtime_buffer_subspan() {
  // CHECK-NEXT: = hal.constant.subspan @storage0_buffer0[#hal.byte_range<0, 1024>] : tensor<4xf32>
  %cst0 = hal.constant.subspan @storage0_buffer0[#hal.byte_range<0, 1024>] : tensor<4xf32>
  // CHECK-NEXT: = hal.constant.subspan @storage0_buffer0[#hal.byte_range<1024, 2048>] : tensor<4xf32>
  %cst1 = hal.constant.subspan @storage0_buffer0[#hal.byte_range<1024, 2048>] : tensor<4xf32>
  return
}
