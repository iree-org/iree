// RUN: iree-opt -split-input-file -iree-spirv-vector-to-cooperative-matrix %s | IreeFileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader, CooperativeMatrixNV, Int8, StorageBuffer8BitAccess], [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix, SPV_KHR_8bit_storage]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  // CHECK-LABEL: func @kernel_matmul
  func @kernel_matmul(%arg0: memref<8x32xi8>, %arg1: memref<32x8xi8>, %arg2: memref<8x8xi32>) attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
    %c0 = constant 0 : index
    %cst = constant 0 : i32
    %cst_i8 = constant 0 : i8
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst_i8 : memref<8x32xi8>, vector<8x32xi8>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst_i8 : memref<32x8xi8>, vector<32x8xi8>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst : memref<8x8xi32>, vector<8x8xi32>
    %3 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} %0, %1, %2 : vector<8x32xi8>, vector<32x8xi8> into vector<8x8xi32>
    vector.transfer_write %3, %arg2[%c0, %c0] : vector<8x8xi32>, memref<8x8xi32>
    // CHECK: %[[A:.+]] = spv.CooperativeMatrixLoadNV
    // CHECK: %[[B:.+]] = spv.CooperativeMatrixLoadNV
    // CHECK: %[[C:.+]] = spv.CooperativeMatrixLoadNV
    // CHECK: %[[R:.+]] = spv.CooperativeMatrixMulAddNV %[[A]], %[[B]], %[[C]]
    // CHECK: spv.CooperativeMatrixStoreNV %{{.*}}, %[[R]], %{{.*}}, %{{.*}}
    return
  }
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader, CooperativeMatrixNV, Int8, Float16, StorageUniform16, StorageBuffer8BitAccess, Float16Buffer], [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix, SPV_KHR_8bit_storage, SPV_KHR_16bit_storage]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  // CHECK-LABEL: func @kernel_matmul_licm
  func @kernel_matmul_licm(%arg0: memref<4096x4096xi8>, %arg1: memref<4096x4096xi8>, %arg2: memref<4096x4096xi32>) attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
    %c32 = constant 32 : index
    %c4096 = constant 4096 : index
    %c0 = constant 0 : index
    %c0_i32 = constant 0 : i32
    %c0_i8 = constant 0 : i8
    // CHECK: %[[C:.+]] = spv.CooperativeMatrixLoadNV
    %4 = vector.transfer_read %arg2[%c0, %c0], %c0_i32 {in_bounds = [true, true]} : memref<4096x4096xi32>, vector<16x16xi32>
    // CHECK: %[[INIT:.+]] = unrealized_conversion_cast %[[C]] : !spv.coopmatrix<16x16xi32, Subgroup> to vector<16x16xi32>
    // CHECK: %[[LOOP:.+]] = scf.for
    // CHECK-SAME: iter_args(%[[ARG:.+]] = %[[INIT]])
    %5 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %4) -> (vector<16x16xi32>) {
      // CHECK: %[[A:.+]] = spv.CooperativeMatrixLoadNV
      %6 = vector.transfer_read %arg0[%c0, %arg3], %c0_i8 {in_bounds = [true, true]} : memref<4096x4096xi8>, vector<16x32xi8>
      // CHECK: %[[B:.+]] = spv.CooperativeMatrixLoadNV
      %7 = vector.transfer_read %arg1[%arg3, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4096x4096xi8>, vector<32x16xi8>
      // CHECK: %[[C1:.+]] = unrealized_conversion_cast %[[ARG]] : vector<16x16xi32> to !spv.coopmatrix<16x16xi32, Subgroup>
      // CHECK: %[[R:.+]] = spv.CooperativeMatrixMulAddNV %[[A]], %[[B]], %[[C1]]
      %8 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} %6, %7, %arg4 : vector<16x32xi8>, vector<32x16xi8> into vector<16x16xi32>
      // CHECK: %[[YIELD:.+]] = unrealized_conversion_cast %[[R]] : !spv.coopmatrix<16x16xi32, Subgroup> to vector<16x16xi32>
      // CHECK: scf.yield %[[YIELD]]
      scf.yield %8 : vector<16x16xi32>
    }
    // CHECK: %[[ACCv:.+]] = unrealized_conversion_cast %[[LOOP]] : vector<16x16xi32> to !spv.coopmatrix<16x16xi32, Subgroup>
    // CHECK: spv.CooperativeMatrixStoreNV %{{.*}}, %[[ACCv]], %{{.*}}, %{{.*}}
    vector.transfer_write %5, %arg2[%c0, %c0] : vector<16x16xi32>, memref<4096x4096xi32>
    return
  }
}

// -----

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader, CooperativeMatrixNV, Int8, Float16, StorageUniform16, StorageBuffer8BitAccess, Float16Buffer], [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix, SPV_KHR_8bit_storage, SPV_KHR_16bit_storage]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  // CHECK-LABEL: func @kernel_matmul_vector_memref
  func @kernel_matmul_vector_memref(%arg0: memref<4096x256xvector<4xi32>>, %arg1: memref<4096x256xvector<4xi32>>, %arg2: memref<4096x1024xvector<4xi32>>) attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
    %c32 = constant 32 : index
    %c4096 = constant 4096 : index
    %c0 = constant 0 : index
    %cst = constant dense<0> : vector<4xi32>
    // CHECK: %[[C:.+]] = spv.CooperativeMatrixLoadNV
    %4 = vector.transfer_read %arg2[%c0, %c0], %cst : memref<4096x1024xvector<4xi32>>, vector<16x16xi32>
    // CHECK: scf.for
    %5 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %4) -> (vector<16x16xi32>) {
      // CHECK: %[[A:.+]] = spv.CooperativeMatrixLoadNV
      %6 = vector.transfer_read %arg0[%c0, %arg3], %cst : memref<4096x256xvector<4xi32>>, vector<16x32xi8>
      // CHECK: %[[B:.+]] = spv.CooperativeMatrixLoadNV
      %7 = vector.transfer_read %arg1[%arg3, %c0], %cst : memref<4096x256xvector<4xi32>>, vector<32x16xi8>
      // CHECK: %[[R:.+]] = spv.CooperativeMatrixMulAddNV %[[A]], %[[B]], %{{.*}}
      %8 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} %6, %7, %arg4 : vector<16x32xi8>, vector<32x16xi8> into vector<16x16xi32>
      scf.yield %8 : vector<16x16xi32>
    }
    // CHECK: spv.CooperativeMatrixStoreNV
    vector.transfer_write %5, %arg2[%c0, %c0] : vector<16x16xi32>, memref<4096x1024xvector<4xi32>>
    return
  }
}
