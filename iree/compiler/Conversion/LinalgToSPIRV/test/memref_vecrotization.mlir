// RUN: iree-opt -split-input-file -iree-spirv-vectorize-memref -canonicalize %s | IreeFileCheck %s

// CHECK-LABEL: func @copy
//  CHECK-SAME: (%[[ARG0:.+]]: memref<4096x1024xvector<4xf32>>
//       CHECK: %[[ALLOC:.+]] = alloc() : memref<128x8xvector<4xf32>, 3>
//       CHECK: %[[V:.+]] = load %[[ARG0]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf32>>
//       CHECK: store %[[V]], %[[ALLOC]][%{{.*}}, %{{.*}}] : memref<128x8xvector<4xf32>, 3>
//       CHECK: %[[MAT:.+]] = vector.transfer_read %[[ARG0]][%{{.*}}, %{{.*}}], %{{.*}} : memref<4096x1024xvector<4xf32>>, vector<32x8xf32>
//       CHECK: vector.transfer_write %[[MAT]], %[[ALLOC]][%{{.*}}, %{{.*}}] : vector<32x8xf32>, memref<128x8xvector<4xf32>, 3>
//       CHECK: dealloc %[[ALLOC]] : memref<128x8xvector<4xf32>, 3>
module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @copy(%arg0: memref<4096x4096xf32>, %x: index, %y: index) attributes {spv.entry_point_abi = {local_size = dense<[128, 1, 1]> : vector<3xi32>}} {
    %cst = constant 0.000000e+00 : f32
    %0 = alloc() : memref<128x32xf32, 3>
    %v = vector.transfer_read %arg0[%x, %y], %cst : memref<4096x4096xf32>, vector<1x4xf32>
    vector.transfer_write %v, %0[%x, %y] : vector<1x4xf32>, memref<128x32xf32, 3>
    %mat = vector.transfer_read %arg0[%x, %y], %cst : memref<4096x4096xf32>, vector<32x8xf32>
    vector.transfer_write %mat, %0[%x, %y] : vector<32x8xf32>, memref<128x32xf32, 3>
    dealloc %0 : memref<128x32xf32, 3>
    return
  }
}

// -----

// Test that the memref is not vectorized if used by scalar load or store.
// CHECK-LABEL: func @copy
//  CHECK-SAME: (%[[ARG0:.+]]: memref<4096x4096xf32>
module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @copy(%arg0: memref<4096x4096xf32>, %x: index, %y: index) attributes {spv.entry_point_abi = {local_size = dense<[128, 1, 1]> : vector<3xi32>}} {
    %cst = constant 0.000000e+00 : f32
    %0 = alloc() : memref<128x32xf32, 3>
    %s = load %arg0[%x, %y] : memref<4096x4096xf32>
    store %s, %0[%x, %y] : memref<128x32xf32, 3>
    dealloc %0 : memref<128x32xf32, 3>
    return
  }
}

// -----

module attributes {spv.target_env = #spv.target_env<#spv.vce<v1.3, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {

// CHECK-LABEL: func @resource_copy
//     CHECK: %[[A:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[B:.+]] = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[V:.+]] = load %[[A]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf32>>
//     CHECK: store %[[V]], %[[B]][%{{.*}}, %{{.*}}] : memref<4096x1024xvector<4xf32>>
//     CHECK: %[[MAT:.+]] = vector.transfer_read %[[A]][%{{.*}}, %{{.*}}], %{{.*}} : memref<4096x1024xvector<4xf32>>, vector<32x8xf32>
//     CHECK: vector.transfer_write %[[MAT]], %[[B]][%{{.*}}, %{{.*}}] {{.*}} : vector<32x8xf32>, memref<4096x1024xvector<4xf32>>
  func @resource_copy() {
    %cst = constant 0.000000e+00 : f32
    %c0 = constant 0 : index
    %0 = iree.placeholder for "interface buffer" {binding = @legacy_io::@arg0} : memref<4096x4096xf32>
    %1 = iree.placeholder for "interface buffer" {binding = @legacy_io::@ret0} : memref<4096x4096xf32>
    %v = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf32>, vector<1x4xf32>
    vector.transfer_write %v, %1[%c0, %c0] : vector<1x4xf32>, memref<4096x4096xf32>
    %mat = vector.transfer_read %0[%c0, %c0], %cst : memref<4096x4096xf32>, vector<32x8xf32>
    vector.transfer_write %mat, %1[%c0, %c0] : vector<32x8xf32>, memref<4096x4096xf32>
    return
  }

  hal.interface @legacy_io attributes {push_constants = 5 : i32, sym_visibility = "private"} {
    hal.interface.binding @arg0, set=1, binding=2, type="StorageBuffer", access="Read"
    hal.interface.binding @ret0, set=3, binding=4, type="StorageBuffer", access="Write"
  }
}

