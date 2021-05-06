// RUN: iree-opt -split-input-file -iree-codegen-vector-to-gpu %s | IreeFileCheck %s

#map0 = affine_map<(d0, d1)[s0] -> (d0 * 4096 + s0 + d1)>

module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @copy(%arg0: memref<4096x4096xf32>) attributes {spv.entry_point_abi = {local_size = dense<[128, 1, 1]> : vector<3xi32>}} {
    %a = memref.alloc() : memref<128x32xf32, 3>
    %c0 = constant 0 : index
    %sv = memref.subview %arg0[%c0, %c0] [128, 32] [1, 1]  : memref<4096x4096xf32> to memref<128x32xf32, #map0>
    linalg.copy(%sv, %a) {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<128x32xf32, #map0>, memref<128x32xf32, 3>
    return
  }
    // CHECK: #[[MAP1:.+]] = affine_map<(d0) -> (d0 * 4)>

    // CHECK-DAG: %[[C1024:.+]] = constant 1024 : index
    // CHECK-DAG: %[[C8:.+]] = constant 8 : index
    // CHECK-DAG: %[[C0:.+]] = constant 0 : index
    // CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<128x32xf32, 3>
    // CHECK: %[[DST:.+]]  = memref.subview %{{.+}}[0, 0] [128, 32] [1, 1]  : memref<4096x4096xf32> to memref<128x32xf32, #map0>
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
    // CHECK:   %[[SIZEx:.+]] = divi_signed %[[IV]], %[[C8]] : index
    // CHECK:   %[[MOD:.+]] = remi_signed %[[IV]], %[[C8]] : index
    // CHECK:   %[[SIZEy:.+]] = affine.apply #[[MAP1]](%[[MOD]])
    // CHECK:   %[[SVs:.+]] = memref.subview %[[DST]][%[[SIZEx]], %[[SIZEy]]] [1, 4] [1, 1]  : memref<128x32xf32, #map0> to memref<1x4xf32
    // CHECK:   %[[SVd:.+]] = memref.subview %[[ALLOC]][%[[SIZEx]], %[[SIZEy]]] [1, 4] [1, 1]  : memref<128x32xf32, 3> to memref<1x4xf32
    // CHECK:   %[[LOAD:.+]] = vector.transfer_read %[[SVs]][%c0, %c0], %cst {{.*}} : memref<1x4xf32, {{.*}}>, vector<1x4xf32>
    // CHECK:   vector.transfer_write %[[LOAD]], %[[SVd]][%[[C0]], %[[C0]]] {{.*}} : vector<1x4xf32>, memref<1x4xf32
}

// -----

module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @extract(%arg0 : vector<1x4xf32>) -> vector<1x1xf32> attributes {spv.entry_point_abi = {local_size = dense<[128, 1, 1]> : vector<3xi32>}} {
    %0 = vector.extract_strided_slice %arg0
      {offsets = [0, 2], sizes = [1, 1], strides = [1, 1]}
        : vector<1x4xf32> to vector<1x1xf32>
    return %0 : vector<1x1xf32>
  }
  // CHECK-LABEL: func @extract
  //  CHECK-SAME: (%[[ARG0:.*]]: vector<1x4xf32>
  //       CHECK:   %[[A:.*]] = vector.extract %[[ARG0]][0, 2] : vector<1x4xf32>
  //       CHECK:   %[[B:.*]] = vector.broadcast %[[A]] : f32 to vector<1x1xf32>
  //       CHECK:   return %[[B]] : vector<1x1xf32>
}
