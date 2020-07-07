// RUN: iree-opt -split-input-file -iree-codegen-convert-to-spirv %s | IreeFileCheck %s

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  // CHECK: spv.globalVariable @__push_constant_var__ : !spv.ptr<!spv.struct<!spv.array<5 x i32, stride=4> [0]>, PushConstant>
  // CHECK: spv.func @push_constant()
  func @push_constant() {
    // CHECK: %[[INDEX_0:.+]] = spv.constant 0 : i32
    // CHECK: %[[INDEX_1:.+]] = spv.constant 2 : i32
    // CHECK: %[[ADDR:.+]] = spv._address_of @__push_constant_var__ : !spv.ptr<!spv.struct<!spv.array<5 x i32, stride=4> [0]>, PushConstant>
    // CHECK: %[[AC:.+]] = spv.AccessChain %[[ADDR]][%[[INDEX_0]], %[[INDEX_1]]] : !spv.ptr<!spv.struct<!spv.array<5 x i32, stride=4> [0]>, PushConstant>
    // CHECK: spv.Load "PushConstant" %[[AC]] : i32
    %0 = hal.interface.load.constant offset = 2 : index
    return
  }

  hal.interface @legacy_io attributes {push_constants = 5 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=0, binding=2, type="StorageBuffer", access="Write"
  }
}

// -----

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {

  // CHECK: spv.globalVariable @__resource_var_3_4__ bind(3, 4) : !spv.ptr<!spv.struct<!spv.array<16 x f32, stride=4> [0]>, StorageBuffer>
  // CHECK: spv.globalVariable @__resource_var_1_2__ bind(1, 2) : !spv.ptr<!spv.struct<!spv.array<16 x f32, stride=4> [0]>, StorageBuffer>
  // CHECK: spv.func @resource_variable()
  func @resource_variable() {
    // CHECK: spv._address_of @__resource_var_1_2__ : !spv.ptr<!spv.struct<!spv.array<16 x f32, stride=4> [0]>, StorageBuffer>
    // CHECK: spv._address_of @__resource_var_3_4__ : !spv.ptr<!spv.struct<!spv.array<16 x f32, stride=4> [0]>, StorageBuffer>
    %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<4x4xf32>
    %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4x4xf32>
    return
  }

  hal.interface @legacy_io attributes {push_constants = 5 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=1, binding=2, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=3, binding=4, type="StorageBuffer", access="Write"
  }
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader, CooperativeMatrixNV, Int8, StorageBuffer8BitAccess], [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix, SPV_KHR_8bit_storage]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  // CHECK: spv.func @kernel_matmul
  func @kernel_matmul(%arg0: memref<8x32xi8>, %arg1: memref<32x8xi8>, %arg2: memref<8x8xi32>) attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
    %c0 = constant 0 : index
    %cst = constant 0 : i32
    %cst_i8 = constant 0 : i8
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst_i8 : memref<8x32xi8>, vector<8x32xi8>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst_i8 : memref<32x8xi8>, vector<32x8xi8>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst : memref<8x8xi32>, vector<8x8xi32>
    %3 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} %0, %1, %2 : vector<8x32xi8>, vector<32x8xi8> into vector<8x8xi32>
    vector.transfer_write %3, %arg2[%c0, %c0] : vector<8x8xi32>, memref<8x8xi32>
    // CHECK: %[[A:.+]] = spv.CooperativeMatrixLoadNV "StorageBuffer" %{{.*}}, %{{.*}}, %{{.*}}
    // CHECK: %[[B:.+]] = spv.CooperativeMatrixLoadNV "StorageBuffer" %{{.*}}, %{{.*}}, %{{.*}}
    // CHECK: %[[C:.+]] = spv.CooperativeMatrixLoadNV "StorageBuffer" %{{.*}}, %{{.*}}, %{{.*}}
    // CHECK: %[[R:.+]] = spv.CooperativeMatrixMulAddNV %[[A]], %[[B]], %[[C]]
    // CHECK: spv.CooperativeMatrixStoreNV "StorageBuffer" %{{.*}}, %[[R]], %{{.*}}, %{{.*}}
    return
  }
}
