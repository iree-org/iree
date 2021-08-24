// RUN: iree-opt -split-input-file -iree-hal-pack-constant-pool-storage -mlir-print-local-scope %s | IreeFileCheck %s

// CHECK-LABEL: hal.constant_pool @pool
hal.constant_pool @pool attributes {
    buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824,
                                                 min_buffer_offset_alignment = 8,
                                                 max_buffer_range = 134217728,
                                                 min_buffer_range_alignment = 4>
  } {
  // CHECK-DAG: hal.constant_pool.span @cst0 : tensor<3xi8> = @_storage[#util.byte_range<0, 3>]
  hal.constant_pool.value @cst0 = dense<[6, 7, 8]> : tensor<3xi8>
  // CHECK-DAG: hal.constant_pool.splat @cst1 = dense<1.000000e+00> : tensor<1xf32>
  hal.constant_pool.value @cst1 = dense<1.000000e+00> : tensor<1xf32>
  // CHECK-DAG: hal.constant_pool.span @cst2 : tensor<4xf32> = @_storage[#util.byte_range<8, 16>]
  hal.constant_pool.value @cst2 = dense<[2.1, 3.2, 4.3, 5.4]> : tensor<4xf32>

  //      CHECK: hal.constant_storage @_storage = #util.composite<24xi8, [
  // CHECK-NEXT:   dense<[6, 7, 8]> : tensor<3xi8>,
  // CHECK-NEXT:   dense<0> : vector<5xi8>,
  // CHECK-NEXT:   dense<[2.100000e+00, 3.200000e+00, 4.300000e+00, 5.400000e+00]> : tensor<4xf32>,
  // CHECK-NEXT: ]>
}

// -----

// CHECK-LABEL: hal.constant_pool @multi_storage
hal.constant_pool @multi_storage attributes {
    buffer_constraints = #hal.buffer_constraints<max_allocation_size = 18,
                                                 min_buffer_offset_alignment = 1,
                                                 max_buffer_range = 134217728,
                                                 min_buffer_range_alignment = 1>
  } {
  // CHECK-DAG: hal.constant_pool.span @cst0 : tensor<4xf32> = @_storage[#util.byte_range<0, 16>]
  hal.constant_pool.value @cst0 = dense<[2.1, 3.2, 4.3, 5.4]> : tensor<4xf32>
  // CHECK-DAG: hal.constant_pool.span @cst1 : tensor<3xi8> = @_storage_0[#util.byte_range<0, 3>]
  hal.constant_pool.value @cst1 = dense<[6, 7, 8]> : tensor<3xi8>

  // CHECK-NEXT: hal.constant_storage @_storage = #util.composite<16xi8, [
  // CHECK-NEXT:   dense<[2.100000e+00, 3.200000e+00, 4.300000e+00, 5.400000e+00]> : tensor<4xf32>,
  // CHECK-NEXT: ]>
  // CHECK-NEXT: hal.constant_storage @_storage_0 = #util.composite<3xi8, [
  // CHECK-NEXT:   dense<[6, 7, 8]> : tensor<3xi8>,
  // CHECK-NEXT: ]>
}
