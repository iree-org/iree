// RUN: iree-opt -pass-pipeline="builtin.module(func.func(iree-llvmgpu-prefetch-shared-memory),cse,canonicalize)" %s | FileCheck %s

// CHECK-LABEL: @prefetch_add
// CHECK-SAME: (%[[GLOBAL:.*]]: memref<128xf32>)
func.func @prefetch_add(%arg0: memref<128xf32>) {
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: %[[C127:.*]] = arith.constant 127 : index
  %c128 = arith.constant 128 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[SHARED:.*]] = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  %alloc = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[PRO_READ:.*]] = vector.transfer_read %[[GLOBAL]]
  // CHECK: vector.transfer_write %[[PRO_READ]], %[[SHARED]]
  // CHECK: %[[OUT:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C127]] step %[[C1]] iter_args(%[[ARG:.*]] = %[[CST]])
  %0 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %cst) -> (vector<1xf32>) {
    // CHECK-DAG: %[[IVPLUS1:.*]] = arith.addi %[[IV]], %[[C1]]
    // CHECK: %[[KER_READ:.*]] = vector.transfer_read %[[GLOBAL]][%[[IVPLUS1]]]
    %1 = vector.transfer_read %arg0[%arg1], %cst_0 : memref<128xf32>, vector<1xf32>
    vector.transfer_write %1, %alloc[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, #gpu.address_space<workgroup>>
    // CHECK: gpu.barrier
    // CHECK: %[[COMPUTE_READ:.*]] = vector.transfer_read %[[SHARED]][%[[C0]]]
    %2 = vector.transfer_read %alloc[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    // CHECK: %[[COMPUTE:.*]] = arith.addf %[[COMPUTE_READ]], %[[ARG]]
    %3 = arith.addf %2, %arg2 : vector<1xf32>
    // CHECK: gpu.barrier
    // CHECK: vector.transfer_write %[[KER_READ]], %[[SHARED]]
    // CHECK: scf.yield %[[COMPUTE]]
    scf.yield %3 : vector<1xf32>
  }
  // CHECK: gpu.barrier
  // CHECK: %[[EPI_READ:.*]] = vector.transfer_read %[[SHARED]][%[[C0]]]
  // CHECK: %[[EPI_COMPUTE:.*]] = arith.addf %[[EPI_READ]], %[[OUT]]
  // CHECK: vector.transfer_write %[[EPI_COMPUTE]], %[[GLOBAL]][%[[C0]]]
  vector.transfer_write %0, %arg0[%c0] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
  return
}

// CHECK-LABEL: @prefetch_multi_scf_return
// CHECK-SAME: (%[[GLOBAL:.*]]: memref<128xf32>)
func.func @prefetch_multi_scf_return(%arg0: memref<128xf32>) -> (vector<1xf32>, vector<1xf32>) {
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: %[[C127:.*]] = arith.constant 127 : index
  %c128 = arith.constant 128 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  // CHECK-DAG: %[[SHARED:.*]] = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  %alloc = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[PRO_READ:.*]] = vector.transfer_read %[[GLOBAL]]
  // CHECK: vector.transfer_write %[[PRO_READ]], %[[SHARED]]
  // CHECK: %[[OUT:.*]]:2 = scf.for %[[IV:.*]] = %[[C0]] to %[[C127]] step %[[C1]] iter_args(%[[ARG:.*]] = %[[CST]], %[[ARG1:.*]] = %[[CST]])
  %0:2 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %cst, %arg3 = %cst) -> (vector<1xf32>, vector<1xf32>) {
    // CHECK-DAG: %[[IVPLUS1:.*]] = arith.addi %[[IV]], %[[C1]]
    // CHECK: %[[KER_READ:.*]] = vector.transfer_read %[[GLOBAL]][%[[IVPLUS1]]]
    %1 = vector.transfer_read %arg0[%arg1], %cst_0 : memref<128xf32>, vector<1xf32>
    vector.transfer_write %1, %alloc[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, #gpu.address_space<workgroup>>
    // CHECK: gpu.barrier
    // CHECK: %[[COMPUTE_READ:.*]] = vector.transfer_read %[[SHARED]][%[[C0]]]
    %2 = vector.transfer_read %alloc[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    // CHECK: %[[COMPUTE:.*]] = arith.addf %[[COMPUTE_READ]], %[[ARG]]
    // CHECK: %[[COMPUTE2:.*]] = arith.addf %[[COMPUTE]], %[[ARG1]]
    %3 = arith.addf %2, %arg2 : vector<1xf32>
    %4 = arith.addf %3, %arg3 : vector<1xf32>
    // CHECK: gpu.barrier
    // CHECK: vector.transfer_write %[[KER_READ]], %[[SHARED]]
    // CHECK: scf.yield %[[COMPUTE]], %[[COMPUTE2]]
    scf.yield %3, %4 : vector<1xf32>, vector<1xf32>
  }
  // CHECK: gpu.barrier
  // CHECK: %[[EPI_READ:.*]] = vector.transfer_read %[[SHARED]][%[[C0]]]
  // CHECK: %[[EPI_COMPUTE:.*]] = arith.addf %[[EPI_READ]], %[[OUT]]#0
  // CHECK: %[[EPI_COMPUTE2:.*]] = arith.addf %[[EPI_COMPUTE]], %[[OUT]]#1
  // CHECK: return %[[EPI_COMPUTE]], %[[EPI_COMPUTE2]]
  return %0#0, %0#1 : vector<1xf32>, vector<1xf32>
}
