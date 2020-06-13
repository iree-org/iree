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
