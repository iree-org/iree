// RUN: iree-opt -pass-pipeline="builtin.module(func.func(iree-llvmgpu-prefetch-shared-memory),cse,canonicalize)" %s --split-input-file | FileCheck %s --check-prefixes=CHECK-ALL,CHECK

// RUN: iree-opt -pass-pipeline="builtin.module(func.func(iree-llvmgpu-prefetch-shared-memory{num-stages=1}))" %s --split-input-file | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-1STAGE
// RUN: iree-opt -pass-pipeline="builtin.module(func.func(iree-llvmgpu-prefetch-shared-memory{num-stages=3}))" %s --split-input-file | FileCheck %s --check-prefixes=CHECK-ALL,CHECK-3STAGE

// CHECK-ALL-LABEL: @prefetch_add
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
  // CHECK-1STAGE: scf.for %{{.*}} = %c0 to %c128 step %c1
  // 3-stage prologue: 2 reads
  // CHECK-3STAGE: vector.transfer_read %arg0
  // CHECK-3STAGE: vector.transfer_write
  // CHECK-3STAGE: vector.transfer_read %arg0
  // CHECK-3STAGE: arith.constant 2 : index
  // CHECK-3STAGE: arith.subi %c128
  // CHECK-3STAGE: scf.for %{{.*}} = %c0 to %{{.*}} step %c1 iter_args
  %0 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %cst) -> (vector<1xf32>) {
    // 2-stage ordering: read -> compute -> write
    // CHECK-DAG: %[[IVPLUS1:.*]] = arith.addi %[[IV]], %[[C1]] : index
    // CHECK: %[[KER_READ:.*]] = vector.transfer_read %[[GLOBAL]][%[[IVPLUS1]]]
    %1 = vector.transfer_read %arg0[%arg1], %cst_0 : memref<128xf32>, vector<1xf32>
    vector.transfer_write %1, %alloc[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, #gpu.address_space<workgroup>>
    // CHECK: gpu.barrier
    // CHECK: %[[COMPUTE_READ:.*]] = vector.transfer_read %[[SHARED]][%[[C0]]]
    %2 = vector.transfer_read %alloc[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    // CHECK: %[[COMPUTE:.*]] = arith.addf %[[COMPUTE_READ]], %[[ARG]]
    %3 = arith.addf %2, %arg2 : vector<1xf32>
    // CHECK: gpu.barrier
    // CHECK: amdgpu.sched_barrier allow = <none>
    // CHECK: vector.transfer_write %[[KER_READ]], %[[SHARED]]
    // CHECK: scf.yield %[[COMPUTE]]

    // 3-stage ordering: compute -> write -> read
    // CHECK-3STAGE: gpu.barrier
    // CHECK-3STAGE: vector.transfer_read %alloc
    // CHECK-3STAGE: arith.addf
    // CHECK-3STAGE: gpu.barrier
    // CHECK-3STAGE: amdgpu.sched_barrier allow = <none>
    // CHECK-3STAGE: vector.transfer_write
    // CHECK-3STAGE: arith.constant 2
    // CHECK-3STAGE: arith.addi
    // CHECK-3STAGE: vector.transfer_read %arg0
    // CHECK-3STAGE: scf.yield
    scf.yield %3 : vector<1xf32>
  }
  // 2-stage epilogue: 1 iteration
  // CHECK: gpu.barrier
  // CHECK: %[[EPI_READ:.*]] = vector.transfer_read %[[SHARED]][%[[C0]]]
  // CHECK: %[[EPI_COMPUTE:.*]] = arith.addf %[[EPI_READ]], %[[OUT]]
  // CHECK: vector.transfer_write %[[EPI_COMPUTE]], %[[GLOBAL]][%[[C0]]]

  // 3-stage epilogue: 2 iterations
  // CHECK-3STAGE: gpu.barrier
  // CHECK-3STAGE: vector.transfer_read %alloc
  // CHECK-3STAGE: arith.addf
  // CHECK-3STAGE: vector.transfer_write
  // CHECK-3STAGE: vector.transfer_read %alloc
  // CHECK-3STAGE: arith.addf
  // CHECK-3STAGE: vector.transfer_write
  vector.transfer_write %0, %arg0[%c0] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
  return
}

// -----

// CHECK-ALL-LABEL: @prefetch_multi_scf_return
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
    // CHECK: amdgpu.sched_barrier allow = <none>
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

// -----

// CHECK-ALL-LABEL: @prefetch_add_with_if
// CHECK-SAME: (%[[GLOBAL:.*]]: memref<128xf32>)
func.func @prefetch_add_with_if(%arg0: memref<128xf32>) {
  // CHECK-DAG: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: %[[C127:.*]] = arith.constant 127 : index
  %c128 = arith.constant 128 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  // CHECK-DAG: %[[SHARED:.*]] = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  %alloc = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  // CHECK-DAG: %[[PRO_READ:.*]] = vector.transfer_read %[[GLOBAL]]
  // CHECK: vector.transfer_write %[[PRO_READ]], %[[SHARED]]
  // CHECK: %[[OUT:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C127]] step %[[C1]] iter_args(%[[ARG:.*]] = %[[CST]])
  %0 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %cst) -> (vector<1xf32>) {
    %dummy = memref.load %arg0[%arg1] : memref<128xf32>
    %5 = arith.cmpf "oeq", %cst_0, %dummy : f32
    // CHECK: %[[BRANCH:.*]] = scf.if %[[COND:.*]] -> (index)
    // CHECK: } else {
    // CHECK: }
    %updated = scf.if %5 -> (index) {
      %override = arith.constant 5 : index
      %add = arith.addi %arg1, %override : index
      scf.yield %add : index
    } else {
      scf.yield %arg1 : index
    }
    // CHECK: %[[KER_READ:.*]] = vector.transfer_read %[[GLOBAL]][%[[UPDATED:.*]]]
    //%1 = vector.transfer_read %arg0[%arg1], %cst_0 : memref<128xf32>, vector<1xf32>
    %1 = vector.transfer_read %arg0[%updated], %cst_0 : memref<128xf32>, vector<1xf32>
    vector.transfer_write %1, %alloc[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, #gpu.address_space<workgroup>>
    // CHECK: gpu.barrier
    // CHECK: %[[COMPUTE_READ:.*]] = vector.transfer_read %[[SHARED]][%[[C0]]]
    %2 = vector.transfer_read %alloc[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    // CHECK: %[[COMPUTE:.*]] = arith.addf %[[COMPUTE_READ]], %[[ARG]]
    %3 = arith.addf %2, %arg2 : vector<1xf32>
    // CHECK: gpu.barrier
    // CHECK: amdgpu.sched_barrier allow = <none>
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

// -----

// CHECK-ALL-LABEL: @noprefetch_copyback
func.func @noprefetch_copyback(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  scf.for %arg2 = %c0 to %c128 step %c1{
    %1 = vector.transfer_read %arg0[%arg2], %cst_0 : memref<128xf32>, vector<1xf32>
    vector.transfer_write %1, %arg1[%arg2] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
  }
  return
}
// CHECK-NOT: gpu.barrier

// -----

// CHECK-ALL-LABEL: @prefetch_scf_if
func.func @prefetch_scf_if(%arg0: memref<128xf32>, %cond : i1) {
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  %0 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %cst) -> (vector<1xf32>) {
    %alloca = memref.alloca() : memref<1xf32, #gpu.address_space<private>>
    scf.if %cond {
      %1 = vector.transfer_read %arg0[%arg1], %cst_0 : memref<128xf32>, vector<1xf32>
      vector.transfer_write %1, %alloca[%c0] : vector<1xf32>, memref<1xf32, #gpu.address_space<private>>
    }
    scf.if %cond {
    %1 = vector.transfer_read %alloca[%c0], %cst_0 : memref<1xf32, #gpu.address_space<private>>, vector<1xf32>
    vector.transfer_write %1, %alloc[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, #gpu.address_space<workgroup>>
    }
    %2 = vector.transfer_read %alloc[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    %3 = arith.addf %2, %arg2 : vector<1xf32>
    scf.yield %3 : vector<1xf32>
  }
  vector.transfer_write %0, %arg0[%c0] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
  return
}

// CHECK-SAME: (%[[GLOBAL:.*]]: memref<128xf32>, %[[COND:.*]]: i1)

// CHECK-DAG: %[[C127:.*]] = arith.constant 127 : index
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
// CHECK-DAG: %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[WG_ALLOC:.*]] = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
// CHECK-DAG: %[[PRIV_ALLOC:.*]] = memref.alloca() : memref<1xf32, #gpu.address_space<private>>

// CHECK: scf.if %[[COND]] {
// CHECK:   %[[IF_READ:.*]] = vector.transfer_read %[[GLOBAL]][%[[C0]]]
// CHECK:   vector.transfer_write %[[IF_READ]], %[[PRIV_ALLOC]][%[[C0]]]
// CHECK:   %[[PRIV_READ:.*]] = vector.transfer_read %[[PRIV_ALLOC]][%[[C0]]]
// CHECK:   vector.transfer_write %[[PRIV_READ]], %[[WG_ALLOC]][%[[C0]]]
// CHECK: }

// CHECK: %[[OUT:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C127]] step %[[C1]] iter_args(%[[ARG:.*]] = %[[CST]])
// CHECK-DAG:   %[[PRIV_ALLOC2:.*]] = memref.alloca() : memref<1xf32, #gpu.address_space<private>>
// CHECK-DAG:   %[[IVP1:.*]] = arith.addi %[[IV]], %[[C1]]
// CHECK:   scf.if %[[COND]] {
// CHECK:     %[[KER_READ:.*]] = vector.transfer_read %[[GLOBAL]][%[[IVP1]]]
// CHECK:     vector.transfer_write %[[KER_READ]], %[[PRIV_ALLOC2]][%[[C0]]]
// CHECK:   }
// CHECK:   gpu.barrier
// CHECK:   %[[COMPUTE_READ:.*]] = vector.transfer_read %[[WG_ALLOC]][%[[C0]]]
// CHECK:   %[[COMPUTE:.*]] = arith.addf %[[COMPUTE_READ]], %[[ARG]]
// CHECK:   gpu.barrier
// CHECK:   amdgpu.sched_barrier allow = <none>
// CHECK:   scf.if %[[COND]] {
// CHECK:     %[[COMPUTE_RELOAD:.*]] = vector.transfer_read %[[PRIV_ALLOC2]][%[[C0]]]
// CHECK:     vector.transfer_write %[[COMPUTE_RELOAD]], %[[WG_ALLOC]][%[[C0]]]
// CHECK:   }
// CHECK: scf.yield %[[COMPUTE]]

// CHECK: gpu.barrier
// CHECK: %[[EPI_READ:.*]] = vector.transfer_read %[[WG_ALLOC]][%[[C0]]]
// CHECK: %[[EPI_COMPUTE:.*]] = arith.addf %[[EPI_READ]], %[[OUT]]
// CHECK: vector.transfer_write %[[EPI_COMPUTE]], %[[GLOBAL]][%[[C0]]]

// -----

// CHECK-ALL-LABEL: @noprefetch_scf_if_readwritetogether
func.func @noprefetch_scf_if_readwritetogether(%arg0: memref<128xf32>, %cond : i1) {
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  %alloca = memref.alloca() : memref<1xf32, #gpu.address_space<private>>
  %0 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %cst) -> (vector<1xf32>) {
    scf.if %cond {
      %1 = vector.transfer_read %arg0[%arg1], %cst_0 : memref<128xf32>, vector<1xf32>
      vector.transfer_write %1, %alloc[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, #gpu.address_space<workgroup>>
    }
    %2 = vector.transfer_read %alloc[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    %3 = arith.addf %2, %arg2 : vector<1xf32>
    scf.yield %3 : vector<1xf32>
  }
  vector.transfer_write %0, %arg0[%c0] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
  return
}

// CHECK-NOT: gpu.barrier

// -----

// CHECK-ALL-LABEL: @noprefetch_unsupportedif
func.func @noprefetch_unsupportedif(%arg0: memref<128xf32>, %cond: i1) {
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  %0 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %cst) -> (vector<1xf32>) {
    scf.if %cond {
      gpu.barrier
    }
    %1 = vector.transfer_read %arg0[%arg1], %cst_0 : memref<128xf32>, vector<1xf32>
    vector.transfer_write %1, %alloc[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, #gpu.address_space<workgroup>>
    %2 = vector.transfer_read %alloc[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    %3 = arith.addf %2, %arg2 : vector<1xf32>
    scf.yield %3 : vector<1xf32>
  }
  vector.transfer_write %0, %arg0[%c0] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
  return
}

// CHECK: gpu.barrier
// CHECK-NOT: gpu.barrier

// -----

// CHECK-ALL-LABEL: @prefetch_scf_if_transientreadwrite
func.func @prefetch_scf_if_transientreadwrite(%arg0: memref<128xf32>, %cond : i1) {
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  %0 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %cst) -> (vector<1xf32>) {
    %alloca = memref.alloca() : memref<1xf32, #gpu.address_space<private>>
    %alloca2 = memref.alloca() : memref<1xf32, #gpu.address_space<private>>
    scf.if %cond {
      %1 = vector.transfer_read %arg0[%arg1], %cst_0 : memref<128xf32>, vector<1xf32>
      vector.transfer_write %1, %alloca[%c0] : vector<1xf32>, memref<1xf32, #gpu.address_space<private>>
    }
    scf.if %cond {
    %1 = vector.transfer_read %alloca[%c0], %cst_0 : memref<1xf32, #gpu.address_space<private>>, vector<1xf32>
    vector.transfer_write %1, %alloca2[%c0] : vector<1xf32>, memref<1xf32, #gpu.address_space<private>>
    }
    %trans_read = vector.transfer_read %alloca2[%c0], %cst_0 : memref<1xf32, #gpu.address_space<private>>, vector<1xf32>
    vector.transfer_write %trans_read, %alloc[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, #gpu.address_space<workgroup>>
    %2 = vector.transfer_read %alloc[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    %3 = arith.addf %2, %arg2 : vector<1xf32>
    scf.yield %3 : vector<1xf32>
  }
  vector.transfer_write %0, %arg0[%c0] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
  return
}
// CHECK-DAG: %[[WG_ALLOC:.*]] = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
// CHECK-DAG: %[[PRIV_ALLOC0:.*]] = memref.alloca() : memref<1xf32, #gpu.address_space<private>>
// CHECK: scf.if
// CHECK: %[[PRIV_ALLOC1:.*]] = memref.alloca() : memref<1xf32, #gpu.address_space<private>>
// CHECK: scf.if
// CHECK: %[[INIT:.*]] = vector.transfer_read %[[PRIV_ALLOC1]]
// CHECK: vector.transfer_write %[[INIT]], %[[WG_ALLOC]]
// CHECK: %[[OUT:.*]] = scf.for
// CHECK:   %[[PRIV_ALLOC2:.*]] = memref.alloca() : memref<1xf32, #gpu.address_space<private>>
// CHECK:   scf.if
// CHECK:   gpu.barrier
// CHECK:   %[[READ3:.*]] = vector.transfer_read %[[WG_ALLOC]]
// CHECK:   %[[COMP:.*]] = arith.addf
// CHECK:   %[[PRIV_ALLOC3:.*]] = memref.alloca() : memref<1xf32, #gpu.address_space<private>>
// CHECK:   scf.if
// CHECK:   %[[READ4:.*]] = vector.transfer_read %[[PRIV_ALLOC3]]
// CHECK:   gpu.barrier
// CHECK:   amdgpu.sched_barrier allow = <none>
// CHECK:   vector.transfer_write %[[READ4]], %[[WG_ALLOC]]
// CHECK:   scf.yield %[[COMP]]

// -----

// Test that nested loops DO get prologue barriers for correctness.
// When a pipelined loop is inside another loop, we need barriers in the
// prologue to prevent data races between iterations of the outer loop.

// CHECK-ALL-LABEL: @prefetch_nested_loop
// CHECK-SAME: (%[[GLOBAL:.*]]: memref<128xf32>)
func.func @prefetch_nested_loop(%arg0: memref<128xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c4 = arith.constant 4 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  // Outer loop - the inner pipelined loop is nested inside this
  // CHECK: scf.for
  scf.for %outer = %c0 to %c4 step %c1 {
    // Inner loop that will be pipelined
    // For nested loops, prologue barriers ARE inserted for correctness.
    // CHECK: vector.transfer_read %[[GLOBAL]]
    // CHECK: gpu.barrier
    // CHECK: vector.transfer_write
    // CHECK: scf.for
    %0 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %cst) -> (vector<1xf32>) {
      %1 = vector.transfer_read %arg0[%arg1], %cst_0 : memref<128xf32>, vector<1xf32>
      vector.transfer_write %1, %alloc[%c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, #gpu.address_space<workgroup>>
      %2 = vector.transfer_read %alloc[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
      %3 = arith.addf %2, %arg2 : vector<1xf32>
      scf.yield %3 : vector<1xf32>
    }
    vector.transfer_write %0, %arg0[%c0] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
  }
  return
}

// -----

// Test case for gather_to_lds with two operands (A and B) - async copy mode.
// This pattern matches matmul-style access:
// - gather_to_lds for operand A (global -> LDS, async)
// - gather_to_lds for operand B (global -> LDS, async)
// - transfer_read A from LDS to register
// - transfer_read B from LDS to register
// - compute (arith.mulf + arith.addf)
// No gpu.barrier needed - vmcnt tracks in-flight memory operations.

// CHECK-ALL-LABEL: @prefetch_gather_to_lds_two_operands
func.func @prefetch_gather_to_lds_two_operands(
    %A_global: memref<128x128xf32>,
    %B_global: memref<128x128xf32>,
    %C_global: memref<128xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index

  // LDS allocations for A and B tiles
  %A_lds = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  %B_lds = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>

  %result = scf.for %k = %c0 to %c128 step %c1 iter_args(%acc = %cst) -> (vector<1xf32>) {
    // Load stage: async gather from global to LDS for both operands
    amdgpu.gather_to_lds %A_global[%c0, %k], %A_lds[%c0] : vector<1xf32>, memref<128x128xf32>, memref<1xf32, #gpu.address_space<workgroup>>
    amdgpu.gather_to_lds %B_global[%k, %c0], %B_lds[%c0] : vector<1xf32>, memref<128x128xf32>, memref<1xf32, #gpu.address_space<workgroup>>

    // Compute stage: read from LDS and accumulate
    %a_val = vector.transfer_read %A_lds[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    %b_val = vector.transfer_read %B_lds[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    %prod = arith.mulf %a_val, %b_val : vector<1xf32>
    %sum = arith.addf %prod, %acc : vector<1xf32>

    scf.yield %sum : vector<1xf32>
  }

  vector.transfer_write %result, %C_global[%c0] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
  return
}

// -----

// Test case: gather_to_lds inside scf.if should NOT be pipelined.
// The pass should leave the loop unchanged because we don't support
// pipelining gather_to_lds operations inside conditional blocks.
// The loop structure should remain: scf.for with scf.if inside.

// CHECK-ALL-LABEL: @gather_to_lds_inside_if_not_pipelined
// CHECK-ALL: scf.for
// CHECK-ALL:   scf.if
// CHECK-ALL:     amdgpu.gather_to_lds
// CHECK-ALL:   vector.transfer_read
// CHECK-ALL:   arith.addf
// CHECK-ALL:   scf.yield
func.func @gather_to_lds_inside_if_not_pipelined(
    %global: memref<128x128xf32>,
    %output: memref<128xf32>,
    %bound: index) {
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index

  %lds = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>

  %result = scf.for %k = %c0 to %c128 step %c1 iter_args(%acc = %cst) -> (vector<1xf32>) {
    // Conditional gather - should prevent pipelining
    %in_bounds = arith.cmpi slt, %k, %bound : index
    scf.if %in_bounds {
      amdgpu.gather_to_lds %global[%c0, %k], %lds[%c0] : vector<1xf32>, memref<128x128xf32>, memref<1xf32, #gpu.address_space<workgroup>>
    }

    // Compute stage
    %val = vector.transfer_read %lds[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    %sum = arith.addf %val, %acc : vector<1xf32>

    scf.yield %sum : vector<1xf32>
  }

  vector.transfer_write %result, %output[%c0] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
  return
}
