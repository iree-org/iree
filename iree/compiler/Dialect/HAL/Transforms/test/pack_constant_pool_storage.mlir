// RUN: iree-opt -split-input-file -iree-hal-pack-constant-pool-storage %s | IreeFileCheck %s

// CHECK-LABEL: hal.constant_pool @pool
hal.constant_pool @pool attributes {
    buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824,
                                                 min_buffer_offset_alignment = 32,
                                                 max_buffer_range = 134217728,
                                                 min_buffer_range_alignment = 4>
  } {
  // CHECK-DAG: hal.constant_pool.splat @cst0 {{.+}} = dense<1.000000e+00> : tensor<1xf32>
  hal.constant_pool.value @cst0 = dense<1.000000e+00> : tensor<1xf32>
  // CHECK-DAG: hal.constant_pool.span @cst1 : tensor<4xf32> {{.+}} = @_storage[#hal.byte_range<0, 16>]
  hal.constant_pool.value @cst1 = dense<[2.1, 3.2, 4.3, 5.4]> : tensor<4xf32>
  // CHECK-DAG: hal.constant_pool.span @cst2 : tensor<3xi8> {{.+}} = @_storage[#hal.byte_range<32, 3>]
  hal.constant_pool.value @cst2 = dense<[6, 7, 8]> : tensor<3xi8>

  // CHECK: hal.constant_storage @_storage {{.+}} = dense<[102, 102, 6, 64, -51, -52, 76, 64, -102, -103, -119, 64, -51, -52, -84, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 8, 0]> : vector<36xi8>
}

// -----

// CHECK-LABEL: hal.constant_pool @multi_storage
hal.constant_pool @multi_storage attributes {
    buffer_constraints = #hal.buffer_constraints<max_allocation_size = 18,
                                                 min_buffer_offset_alignment = 1,
                                                 max_buffer_range = 134217728,
                                                 min_buffer_range_alignment = 1>
  } {
  // CHECK-DAG: hal.constant_pool.span @cst0 : tensor<4xf32> {{.+}} = @_storage[#hal.byte_range<0, 16>]
  hal.constant_pool.value @cst0 = dense<[2.1, 3.2, 4.3, 5.4]> : tensor<4xf32>
  // CHECK-DAG: hal.constant_pool.span @cst1 : tensor<3xi8> {{.+}} = @_storage_0[#hal.byte_range<0, 3>]
  hal.constant_pool.value @cst1 = dense<[6, 7, 8]> : tensor<3xi8>

  // CHECK-NEXT: hal.constant_storage @_storage {{.+}} = dense<[102, 102, 6, 64, -51, -52, 76, 64, -102, -103, -119, 64, -51, -52, -84, 64]> : vector<16xi8>
  // CHECK-NEXT: hal.constant_storage @_storage_0 {{.+}} = dense<[6, 7, 8]> : vector<3xi8>
}
