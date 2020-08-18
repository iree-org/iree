// RUN: iree-opt -split-input-file -iree-codegen-vector-to-gpu %s | IreeFileCheck %s

module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @kernel_matmul(%arg0: memref<32xf32>, %arg1: memref<32xf32>, %arg2: memref<32xf32>) attributes {spv.entry_point_abi = {local_size = dense<[32, 1, 1]> : vector<3xi32>}} {
    %c0 = constant 0 : index
    %cst = constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0], %cst : memref<32xf32>, vector<32xf32>
    %1 = vector.transfer_read %arg1[%c0], %cst : memref<32xf32>, vector<32xf32>
    %2 = addf %0, %1 : vector<32xf32>
    vector.transfer_write %2, %arg2[%c0] : vector<32xf32>, memref<32xf32>
    return
  }
  // CHECK: %[[C0:.+]] = constant 0 : index
  // CHECK: %[[TId:.+]] = "gpu.thread_id"() {dimension = "x"} : () -> index
  // CHECK: %[[Index:.+]] = addi %[[TId]], %[[C0]] : index
  // CHECK-DAG: %[[A:.+]] = load %arg0[%[[Index]]] : memref<32xf32>
  // CHECK-DAG: %[[B:.+]] = load %arg1[%{{.*}}] : memref<32xf32>
  // CHECK: %[[C:.+]] = addf %[[A]], %[[B]] : f32
  // CHECK: store %[[C]], %arg2[%{{.*}}] : memref<32xf32>
}

// -----

#map0 = affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>

module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @copy(%arg0: memref<4096x4096xf32>) attributes {spv.entry_point_abi = {local_size = dense<[128, 1, 1]> : vector<3xi32>}} {
    %a = alloc() : memref<128x32xf32, 3>
    %c0 = constant 0 : index
    %sv = subview %arg0[%c0, %c0] [128, 32] [1, 1]  : memref<4096x4096xf32> to memref<128x32xf32, #map0>
    linalg.copy(%sv, %a) {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<128x32xf32, #map0>, memref<128x32xf32, 3>
    return
  }
    // CHECK: #[[MAP1:.+]] = affine_map<(d0) -> (d0 * 4 - (d0 floordiv 8) * 32)>

    // CHECK: %[[C1024:.+]] = constant 1024 : index
    // CHECK: %[[C8:.+]] = constant 8 : index
    // CHECK: %[[C0:.+]] = constant 0 : index
    // CHECK: %[[ALLOC:.+]] = alloc() : memref<128x32xf32, 3>
    // CHECK: %[[DST:.+]]  = subview %{{.+}}[0, 0] [128, 32] [1, 1]  : memref<4096x4096xf32> to memref<128x32xf32, #map0>
    // CHECK: %[[TIDx:.+]] = "gpu.thread_id"() {dimension = "x"} : () -> index
    // CHECK: %[[DIMx:.+]] = "gpu.block_dim"() {dimension = "x"} : () -> index
    // CHECK: %[[TIDy:.+]] = "gpu.thread_id"() {dimension = "y"} : () -> index
    // CHECK: %[[DIMy:.+]] = "gpu.block_dim"() {dimension = "y"} : () -> index
    // CHECK: %[[TIDz:.+]] = "gpu.thread_id"() {dimension = "z"} : () -> index
    // CHECK: %[[DIMz:.+]] = "gpu.block_dim"() {dimension = "z"} : () -> index
    // CHECK: %[[LIDz:.+]] = muli %[[TIDz]], %[[DIMy]] : index
    // CHECK: %[[LIDzy:.+]] = addi %[[LIDz]], %[[TIDy]] : index
    // CHECK: %[[DIMzy:.+]] = muli %[[DIMz]], %[[DIMy]] : index
    // CHECK: %[[LIDzyx:.+]] = muli %[[LIDzy]], %[[DIMx]] : index
    // CHECK: %[[LID:.+]] = addi %[[LIDzyx]], %[[TIDx]] : index
    // CHECK: %[[DIMzyx:.+]] = muli %[[DIMzy]], %[[DIMx]] : index
    // CHECK: scf.for %[[IV:.+]] = %[[LID]] to %[[C1024]] step %[[DIMzyx]] {
      // CHECK: %[[SIZEx:.+]] = divi_signed %[[IV]], %[[C8]] : index
      // CHECK: %[[SIZEy:.+]] = affine.apply #[[MAP1]](%[[IV]])
      // CHECK: %[[SVs:.+]] = subview %[[DST]][%[[SIZEx]], %[[SIZEy]]] [1, 4] [1, 1]  : memref<128x32xf32, #map0> to memref<1x4xf32
      // CHECK: %[[SVd:.+]] = subview %[[ALLOC]][%[[SIZEx]], %[[SIZEy]]] [1, 4] [1, 1]  : memref<128x32xf32, 3> to memref<1x4xf32
      // CHECK: %[[LOAD:.+]] = vector.transfer_read %[[SVs]][%c0, %c0], %cst {{.*}} : memref<1x4xf32, {{.*}}>, vector<1x4xf32>
      // CHECK: vector.transfer_write %[[LOAD]], %[[SVd]][%[[C0]], %[[C0]]] {{.*}} : vector<1x4xf32>, memref<1x4xf32
}
