module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @reshape_1D_2D_ex_dispatch_0() {
    %c0 = constant 0 : index
    %0 = hal.interface.load.tensor @legacy_io::@arg0, offset = %c0 : tensor<12xf32>
    %1 = alloc() : memref<12xf32>
    tensor_store %0, %1 : memref<12xf32>
    %2 = alloc() : memref<3x4xf32>
    call @reshape_1D_2D_ex_dispatch_0_impl(%1, %2) : (memref<12xf32>, memref<3x4xf32>) -> ()
    %3 = tensor_load %2 : memref<3x4xf32>
    hal.interface.store.tensor %3, @legacy_io::@ret0, offset = %c0 : tensor<3x4xf32>
    return
  }
  func @reshape_1D_2D_ex_dispatch_0_impl(%arg0: memref<12xf32> {spv.interface_var_abi = {binding = 0 : i32, descriptor_set = 0 : i32}}, %arg1: memref<3x4xf32> {spv.interface_var_abi = {binding = 1 : i32, descriptor_set = 0 : i32}}) attributes {iree.dispatch_fn_name = "reshape_1D_2D_ex_dispatch_0", spv.entry_point_abi = {local_size = dense<1> : vector<3xi32>}} {
    %0 = iree.load_input(%arg0 : memref<12xf32>) : tensor<12xf32>
    %1 = "xla_hlo.reshape"(%0) : (tensor<12xf32>) -> tensor<3x4xf32>
    iree.store_output(%1 : tensor<3x4xf32>, %arg1 : memref<3x4xf32>)
    return
  }
  hal.interface @legacy_io attributes {sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=1, type="StorageBuffer", access="Write|Discard"
  }
}