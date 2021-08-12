// RUN: iree-opt -split-input-file -iree-hal-identify-constant-pools %s | IreeFileCheck %s

#device_target_cpu = #hal.device.target<"cpu", {
  buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824, min_buffer_offset_alignment = 16, max_buffer_range = 1073741824, min_buffer_range_alignment = 16>
}>
#device_target_gpu = #hal.device.target<"gpu", {
  buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824, min_buffer_offset_alignment = 256, max_buffer_range = 134217728, min_buffer_range_alignment = 16>
}>
module attributes {
  hal.device.targets = [#device_target_cpu, #device_target_gpu]
} {

//      CHECK: hal.constant_pool @_const_pool attributes
// CHECK-SAME:     buffer_constraints = #hal.buffer_constraints<max_allocation_size = 1073741824, min_buffer_offset_alignment = 256, max_buffer_range = 134217728, min_buffer_range_alignment = 16>
// CHECK-NEXT:   hal.constant_pool.value @cst0 = dense<1.000000e+00> : tensor<1xf32>
util.global private @cst0 = dense<1.000000e+00> : tensor<1xf32>
// CHECK-NEXT:   hal.constant_pool.value @cst1 = dense<[2.100000e+00, 3.200000e+00, 4.300000e+00, 5.400000e+00]> : tensor<4xf32>
util.global private @cst1 = dense<[2.1, 3.2, 4.3, 5.4]> : tensor<4xf32>
// CHECK-NEXT:   hal.constant_pool.value @cst2 = dense<[6, 7, 8]> : tensor<3xi8>
util.global private @cst2 = dense<[6, 7, 8]> : tensor<3xi8>

// CHECK-LABEL: func @immutable_variables
func @immutable_variables() -> (tensor<1xf32>, tensor<4xf32>, tensor<3xi8>) {
  // CHECK-NEXT: = hal.constant_pool.load @_const_pool::@cst0 : tensor<1xf32>
  %cst0 = util.global.load @cst0 : tensor<1xf32>
  // CHECK-NEXT: = hal.constant_pool.load @_const_pool::@cst1 : tensor<4xf32>
  %cst1 = util.global.load @cst1 : tensor<4xf32>
  // CHECK-NEXT: = hal.constant_pool.load @_const_pool::@cst2 : tensor<3xi8>
  %cst2 = util.global.load @cst2 : tensor<3xi8>
  return %cst0, %cst1, %cst2 : tensor<1xf32>, tensor<4xf32>, tensor<3xi8>
}

}

// -----

//      CHECK: hal.constant_pool @_const_pool_init
// CHECK-NEXT:   hal.constant_pool.value @variable_0 = dense<3.000000e+00> : tensor<128xf32>

// CHECK: util.global private mutable @variable_0 initializer(@variable_0_initializer)
util.global private mutable @variable_0 = dense<3.0> : tensor<128xf32>
// CHECK-NEXT: func private @variable_0_initializer() -> tensor<128xf32>
// CHECK-NEXT:   [[CONST:%.+]] = hal.constant_pool.load @_const_pool_init::@variable_0 : tensor<128xf32>
// CHECK-NEXT:   return [[CONST]] : tensor<128xf32>
// CHECK-NEXT: }

// CHECK-LABEL: func @mutable_variables
func @mutable_variables() -> tensor<128xf32> {
  // CHECK: util.global.load @variable_0
  %var_0 = util.global.load @variable_0 : tensor<128xf32>
  return %var_0 : tensor<128xf32>
}

// -----

// NOTE: indirect variable accesses not currently supported.
// CHECK: util.global private @_large_const_0
util.global private @_large_const_0 = dense<3.0> : tensor<128xf32>
func @skip_indirect_variables() -> (tensor<128xf32>) {
  // CHECK: util.global.address
  %0 = util.global.address @_large_const_0 : !util.ptr<tensor<128xf32>>
  // CHECK: util.global.load.indirect
  %1 = util.global.load.indirect %0 : !util.ptr<tensor<128xf32>> -> tensor<128xf32>
  return %1 : tensor<128xf32>
}
