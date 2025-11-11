// RUN: iree-opt -pass-pipeline="builtin.module(func.func(iree-llvmgpu-prefetch-shared-memory{num-stages=2}),cse,canonicalize)" %s --split-input-file | FileCheck %s --check-prefixes=CHECK-ALL,CHECK
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
  // CHECK: gpu.barrier
  // CHECK: vector.transfer_write %[[PRO_READ]], %[[SHARED]]
  // CHECK: %[[OUT:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C127]] step %[[C1]] iter_args(%[[ARG:.*]] = %[[CST]])
  // CHECK-1STAGE: scf.for %{{.*}} = %c0 to %c128 step %c1
  // 3-stage without MFMA: still pipelines but shared read not prefetched (stays in stage 2 with compute)
  // CHECK-3STAGE: vector.transfer_read %arg0
  // CHECK-3STAGE: gpu.barrier
  // CHECK-3STAGE: vector.transfer_write {{.*}}, %alloc
  // CHECK-3STAGE: vector.transfer_read %arg0
  // CHECK-3STAGE: arith.constant 2
  // CHECK-3STAGE: arith.subi %c128
  // CHECK-3STAGE: scf.for %{{.*}} = %c0 to %{{.*}} step %c1
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
    // 3-stage loop body: shared read in stage 2 (with compute), then barrier, then write
    // CHECK-3STAGE: gpu.barrier
    // CHECK-3STAGE: vector.transfer_read %alloc
    // CHECK-3STAGE: arith.addf
    // CHECK-3STAGE: gpu.barrier
    // CHECK-3STAGE: amdgpu.sched_barrier
    // CHECK-3STAGE: vector.transfer_write {{.*}}, %alloc
    scf.yield %3 : vector<1xf32>
  }
  // 2-stage epilogue: 1 iteration
  // CHECK: gpu.barrier
  // CHECK: %[[EPI_READ:.*]] = vector.transfer_read %[[SHARED]][%[[C0]]]
  // CHECK: %[[EPI_COMPUTE:.*]] = arith.addf %[[EPI_READ]], %[[OUT]]
  // CHECK: vector.transfer_write %[[EPI_COMPUTE]], %[[GLOBAL]][%[[C0]]]
  // 3-stage epilogue
  // CHECK-3STAGE: gpu.barrier
  // CHECK-3STAGE: vector.transfer_read %alloc
  // CHECK-3STAGE: arith.addf
  // CHECK-3STAGE: vector.transfer_write {{.*}}, %arg0
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
  // CHECK: gpu.barrier
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
  // CHECK: gpu.barrier
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
// CHECK: }
// CHECK: gpu.barrier
// CHECK: scf.if %[[COND]] {
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
// CHECK: gpu.barrier
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

// CHECK-ALL-LABEL: @prefetch_mfma
// CHECK-SAME: (%[[A:.*]]: memref<128xbf16>, %[[B:.*]]: memref<128xbf16>)
func.func @prefetch_mfma(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>) {
  %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
  %cst_0 = arith.constant 0.000000e+00 : bf16
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc_a = memref.alloc() : memref<4xbf16, #gpu.address_space<workgroup>>
  %alloc_b = memref.alloc() : memref<4xbf16, #gpu.address_space<workgroup>>

  // 3-stage with MFMA: prologue includes BOTH shared reads for first MFMA
  // Prologue: global reads, barrier, shared writes, BOTH shared reads, global reads
  // CHECK-3STAGE: vector.transfer_read %arg0
  // CHECK-3STAGE: vector.transfer_read %arg1
  // CHECK-3STAGE: gpu.barrier
  // CHECK-3STAGE: vector.transfer_write {{.*}}, %alloc
  // CHECK-3STAGE: vector.transfer_write {{.*}}, %alloc
  // CHECK-3STAGE: vector.transfer_read %alloc
  // CHECK-3STAGE: vector.transfer_read %alloc
  // CHECK-3STAGE: vector.transfer_read %arg0
  // CHECK-3STAGE: vector.transfer_read %arg1
  // CHECK-3STAGE: scf.for
  %0 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %cst) -> (vector<4xf32>) {
    %1 = vector.transfer_read %arg0[%arg2], %cst_0 : memref<128xbf16>, vector<4xbf16>
    vector.transfer_write %1, %alloc_a[%c0] {in_bounds = [true]} : vector<4xbf16>, memref<4xbf16, #gpu.address_space<workgroup>>
    %2 = vector.transfer_read %arg1[%arg2], %cst_0 : memref<128xbf16>, vector<4xbf16>
    vector.transfer_write %2, %alloc_b[%c0] {in_bounds = [true]} : vector<4xbf16>, memref<4xbf16, #gpu.address_space<workgroup>>
    // Read both operands from shared memory
    %3 = vector.transfer_read %alloc_a[%c0], %cst_0 : memref<4xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
    %4 = vector.transfer_read %alloc_b[%c0], %cst_0 : memref<4xbf16, #gpu.address_space<workgroup>>, vector<4xbf16>
    // MFMA operation with two shared memory operands
    %5 = amdgpu.mfma 16x16x16 %3 * %4 + %arg3 { abid = 0 : i32, cbsz = 0 : i32, blocks = 1 : i32 } blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>
    // Inside loop body: MFMA uses BOTH operands from prev iter, then barriers, writes, reads
    // All shared reads for first compute are prefetched, so MFMA executes immediately
    // CHECK-3STAGE: amdgpu.mfma
    // CHECK-3STAGE: gpu.barrier
    // CHECK-3STAGE: amdgpu.sched_barrier
    // CHECK-3STAGE: vector.transfer_write {{.*}}, %alloc
    // CHECK-3STAGE: vector.transfer_write {{.*}}, %alloc
    // CHECK-3STAGE: gpu.barrier
    // CHECK-3STAGE: vector.transfer_read %alloc
    // CHECK-3STAGE: vector.transfer_read %alloc
    scf.yield %5 : vector<4xf32>
  }
  return
}

// -----

// Test MFMA with intermediate vector operations (transpose, extract, shape_cast)
// This pattern is common in real matmul kernels where shared memory reads
// go through transformations before reaching MFMA operations.
// CHECK-ALL-LABEL: @prefetch_mfma_with_transforms
func.func @prefetch_mfma_with_transforms(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>) {
  %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
  %cst_0 = arith.constant 0.000000e+00 : bf16
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc_a = memref.alloc() : memref<2x4xbf16, #gpu.address_space<workgroup>>
  %alloc_b = memref.alloc() : memref<2x4xbf16, #gpu.address_space<workgroup>>

  // 3-stage: Prologue should prefetch BOTH shared reads that feed the first MFMA
  // Even though there are intermediate transforms (transpose, extract, shape_cast),
  // the prefetch pass should trace back through all the vector ops to find the shared reads
  // CHECK-3STAGE: vector.transfer_read %arg0
  // CHECK-3STAGE: vector.transfer_read %arg1
  // CHECK-3STAGE: gpu.barrier
  // CHECK-3STAGE: vector.transfer_write {{.*}}, %alloc
  // CHECK-3STAGE: vector.transfer_write {{.*}}, %alloc
  // CHECK-3STAGE: vector.transfer_read %alloc{{.*}} : memref<2x4xbf16, #gpu.address_space<workgroup>>, vector<2x4xbf16>
  // CHECK-3STAGE: vector.transfer_read %alloc{{.*}} : memref<2x4xbf16, #gpu.address_space<workgroup>>, vector<2x4xbf16>
  // CHECK-3STAGE: vector.transfer_read %arg0
  // CHECK-3STAGE: vector.transfer_read %arg1
  // CHECK-3STAGE: scf.for
  %0 = scf.for %arg2 = %c0 to %c32 step %c1 iter_args(%arg3 = %cst) -> (vector<4xf32>) {
    // Load from global and write to shared memory
    %1 = vector.transfer_read %arg0[%arg2], %cst_0 : memref<128xbf16>, vector<8xbf16>
    %2 = vector.shape_cast %1 : vector<8xbf16> to vector<2x4xbf16>
    vector.transfer_write %2, %alloc_a[%c0, %c0] {in_bounds = [true, true]} : vector<2x4xbf16>, memref<2x4xbf16, #gpu.address_space<workgroup>>
    %3 = vector.transfer_read %arg1[%arg2], %cst_0 : memref<128xbf16>, vector<8xbf16>
    %4 = vector.shape_cast %3 : vector<8xbf16> to vector<2x4xbf16>
    vector.transfer_write %4, %alloc_b[%c0, %c0] {in_bounds = [true, true]} : vector<2x4xbf16>, memref<2x4xbf16, #gpu.address_space<workgroup>>

    // Read from shared memory with transformations before MFMA
    %5 = vector.transfer_read %alloc_a[%c0, %c0], %cst_0 : memref<2x4xbf16, #gpu.address_space<workgroup>>, vector<2x4xbf16>
    %6 = vector.transpose %5, [1, 0] : vector<2x4xbf16> to vector<4x2xbf16>
    %7 = vector.extract %6[0] : vector<2xbf16> from vector<4x2xbf16>
    %8 = vector.shape_cast %7 : vector<2xbf16> to vector<2xbf16>
    %9 = vector.extract %6[1] : vector<2xbf16> from vector<4x2xbf16>

    %10 = vector.transfer_read %alloc_b[%c0, %c0], %cst_0 : memref<2x4xbf16, #gpu.address_space<workgroup>>, vector<2x4xbf16>
    %11 = vector.transpose %10, [1, 0] : vector<2x4xbf16> to vector<4x2xbf16>
    %12 = vector.extract %11[0] : vector<2xbf16> from vector<4x2xbf16>
    %13 = vector.shape_cast %12 : vector<2xbf16> to vector<2xbf16>
    %14 = vector.extract %11[1] : vector<2xbf16> from vector<4x2xbf16>

    // MFMA operations using transformed vectors
    // The prefetch pass should trace back through all the transforms to find the shared reads
    %15 = vector.shape_cast %8 : vector<2xbf16> to vector<2xbf16>
    %16 = vector.shape_cast %13 : vector<2xbf16> to vector<2xbf16>
    %17 = builtin.unrealized_conversion_cast %15 : vector<2xbf16> to vector<4xbf16>
    %18 = builtin.unrealized_conversion_cast %16 : vector<2xbf16> to vector<4xbf16>
    %19 = amdgpu.mfma 16x16x16 %17 * %18 + %arg3 blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>

    %20 = vector.shape_cast %9 : vector<2xbf16> to vector<2xbf16>
    %21 = vector.shape_cast %14 : vector<2xbf16> to vector<2xbf16>
    %22 = builtin.unrealized_conversion_cast %20 : vector<2xbf16> to vector<4xbf16>
    %23 = builtin.unrealized_conversion_cast %21 : vector<2xbf16> to vector<4xbf16>
    %24 = amdgpu.mfma 16x16x16 %22 * %23 + %19 blgp =  none : vector<4xbf16>, vector<4xbf16>, vector<4xf32>

    // CHECK-3STAGE: amdgpu.mfma
    // CHECK-3STAGE: amdgpu.mfma
    // CHECK-3STAGE: gpu.barrier
    // CHECK-3STAGE: amdgpu.sched_barrier
    // CHECK-3STAGE: vector.transfer_write
    // CHECK-3STAGE: vector.transfer_write
    // CHECK-3STAGE: gpu.barrier
    // CHECK-3STAGE: vector.transfer_read %alloc{{.*}} : memref<2x4xbf16, #gpu.address_space<workgroup>>, vector<2x4xbf16>
    // CHECK-3STAGE: vector.transfer_read %alloc{{.*}} : memref<2x4xbf16, #gpu.address_space<workgroup>>, vector<2x4xbf16>
    scf.yield %24 : vector<4xf32>
  }
  return
}
