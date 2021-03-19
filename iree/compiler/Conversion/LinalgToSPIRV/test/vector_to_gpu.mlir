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
  // CHECK-DAG: %[[A:.+]] = memref.load %arg0[%[[Index]]] : memref<32xf32>
  // CHECK-DAG: %[[B:.+]] = memref.load %arg1[%{{.*}}] : memref<32xf32>
  // CHECK: %[[C:.+]] = addf %[[A]], %[[B]] : f32
  // CHECK: memref.store %[[C]], %arg2[%{{.*}}] : memref<32xf32>
}

// -----

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

    // CHECK: %[[C1024:.+]] = constant 1024 : index
    // CHECK: %[[C8:.+]] = constant 8 : index
    // CHECK: %[[C0:.+]] = constant 0 : index
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
  func @transfer_ops(%arg0: memref<32x32xf32>, %arg1 : vector<1x1xf32>) -> vector<1x1xf32> attributes {spv.entry_point_abi = {local_size = dense<[128, 1, 1]> : vector<3xi32>}} {
  %c0 = constant 0 : index
  %cst = constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<1x1xf32>
  vector.transfer_write %arg1, %arg0[%c0, %c0] : vector<1x1xf32>, memref<32x32xf32>
  return %0 : vector<1x1xf32>
  }
  // CHECK-LABEL: func @transfer_ops
  //  CHECK-SAME: (%[[ARG0:.*]]: memref<32x32xf32>, %[[ARG1:.*]]: vector<1x1xf32>
  //       CHECK:   %[[C0:.*]] = constant 0 : index
  //       CHECK:   %[[LOAD:.*]] = memref.load %[[ARG0]][%[[C0]], %[[C0]]] : memref<32x32xf32>
  //       CHECK:   %[[B:.*]] = vector.broadcast %[[LOAD]] : f32 to vector<1x1xf32>
  //       CHECK:   %[[EXT:.*]] = vector.extract %[[ARG1]][0, 0] : vector<1x1xf32>
  //       CHECK:   memref.store %[[EXT]], %[[ARG0]][%[[C0]], %[[C0]]] : memref<32x32xf32>
  //       CHECK:   return %[[B]] : vector<1x1xf32>
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

module attributes {gpu.container_module, spv.target_env = #spv.target_env<#spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {max_compute_workgroup_invocations = 128 : i32, max_compute_workgroup_size = dense<[128, 128, 64]> : vector<3xi32>}>} {
  func @contract_ops(%arg0 : vector<1x1xf32>, %arg1 : vector<1x4xf32>,
                    %arg2 : vector<1x4xf32>, %arg3 : vector<1x1xf32>,
                    %arg4 : vector<1x1xf32>) -> (vector<1x1xf32>, vector<1x4xf32>) attributes {spv.entry_point_abi = {local_size = dense<[128, 1, 1]> : vector<3xi32>}} {
  %0 = vector.contract {indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]} %arg0, %arg3, %arg4
                      : vector<1x1xf32>, vector<1x1xf32> into vector<1x1xf32>
  %1 = vector.contract {indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "reduction"]} %arg0, %arg1, %arg2
                      : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
  return %0, %1 : vector<1x1xf32>, vector<1x4xf32>
  }
  // CHECK-LABEL: func @contract_ops
  //  CHECK-SAME: (%[[ARG0:.*]]: vector<1x1xf32>, %[[ARG1:.*]]: vector<1x4xf32>, %[[ARG2:.*]]: vector<1x4xf32>, %[[ARG3:.*]]: vector<1x1xf32>, %[[ARG4:.*]]: vector<1x1xf32>)
  //       CHECK:   %[[A:.*]] = vector.extract %[[ARG0]][0, 0] : vector<1x1xf32>
  //       CHECK:   %[[B:.*]] = vector.extract %[[ARG3]][0, 0] : vector<1x1xf32>
  //       CHECK:   %[[C:.*]] = vector.extract %[[ARG4]][0, 0] : vector<1x1xf32>
  //       CHECK:   %[[MUL:.*]] = mulf %[[A]], %[[B]] : f32
  //       CHECK:   %[[ADD:.*]] = addf %[[MUL]], %[[C]] : f32
  //       CHECK:   %[[R0:.*]] = vector.broadcast %[[ADD]] : f32 to vector<1x1xf32>
  //       CHECK:   %[[A:.*]] = vector.extract %[[ARG0]][0, 0] : vector<1x1xf32>
  //       CHECK:   %[[VA:.*]] = vector.broadcast %[[A]] : f32 to vector<4xf32>
  //       CHECK:   %[[VB:.*]] = vector.shape_cast %[[ARG1]] : vector<1x4xf32> to vector<4xf32>
  //       CHECK:   %[[VC:.*]] = vector.shape_cast %[[ARG2]] : vector<1x4xf32> to vector<4xf32>
  //       CHECK:   %[[VMUL:.*]] = mulf %[[VA]], %[[VB]] : vector<4xf32>
  //       CHECK:   %[[VADD:.*]] = addf %[[VMUL]], %[[VC]] : vector<4xf32>
  //       CHECK:   %[[R1:.*]] = vector.shape_cast %[[VADD]] : vector<4xf32> to vector<1x4xf32>
  //       CHECK:   return %[[R0]], %[[R1]] : vector<1x1xf32>, vector<1x4xf32>
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
