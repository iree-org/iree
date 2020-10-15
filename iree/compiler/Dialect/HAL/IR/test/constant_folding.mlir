// RUN: iree-opt -split-input-file -canonicalize %s | iree-opt -split-input-file | IreeFileCheck %s

hal.constant_pool @pool attributes {
    buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824,
                                                 min_buffer_offset_alignment = 32,
                                                 max_buffer_range = 134217728,
                                                 min_buffer_range_alignment = 4>
  } {
  hal.constant_pool.span @cst_span : tensor<3xi8> = @_storage1[#hal.byte_range<0, 3>] -> @pool_storage1_buffer[#hal.byte_range<0, 3>]
  hal.constant_pool.splat @cst_splat = dense<1.000000e+00> : tensor<1xf32> -> @pool_splats[#hal.byte_range<0, 4>]
}

// CHECK-LABEL: func @pools_identified
func @pools_identified() -> (tensor<2x3xf32>, tensor<3x2xf32>) {
  // CHECK-NEXT: = hal.constant.subspan @pool_storage1_buffer[#hal.byte_range<0, 3>] : tensor<2x3xf32>
  %cst0 = hal.constant_pool.load @pool::@cst_span : tensor<2x3xf32>
  // CHECK-NEXT: = hal.constant.subspan @pool_splats[#hal.byte_range<0, 4>] : tensor<3x2xf32>
  %cst1 = hal.constant_pool.load @pool::@cst_splat : tensor<3x2xf32>
  return %cst0, %cst1 : tensor<2x3xf32>, tensor<3x2xf32>
}
