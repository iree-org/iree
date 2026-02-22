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
    // CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
    // CHECK: %[[COMPUTE_READ:.*]] = vector.transfer_read %[[SHARED]][%[[C0]]]
    %2 = vector.transfer_read %alloc[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    // CHECK: %[[COMPUTE:.*]] = arith.addf %[[COMPUTE_READ]], %[[ARG]]
    %3 = arith.addf %2, %arg2 : vector<1xf32>
    // CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
    // CHECK: amdgpu.sched_barrier allow = <none>
    // CHECK: vector.transfer_write %[[KER_READ]], %[[SHARED]]
    // CHECK: scf.yield %[[COMPUTE]]

    // 3-stage ordering: compute -> write -> read
    // CHECK-3STAGE: gpu.barrier memfence [#gpu.address_space<workgroup>]
    // CHECK-3STAGE: vector.transfer_read %alloc
    // CHECK-3STAGE: arith.addf
    // CHECK-3STAGE: gpu.barrier memfence [#gpu.address_space<workgroup>]
    // CHECK-3STAGE: amdgpu.sched_barrier allow = <none>
    // CHECK-3STAGE: vector.transfer_write
    // CHECK-3STAGE: arith.constant 2
    // CHECK-3STAGE: arith.addi
    // CHECK-3STAGE: vector.transfer_read %arg0
    // CHECK-3STAGE: scf.yield
    scf.yield %3 : vector<1xf32>
  }
  // 2-stage epilogue: 1 iteration
  // CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
  // CHECK: %[[EPI_READ:.*]] = vector.transfer_read %[[SHARED]][%[[C0]]]
  // CHECK: %[[EPI_COMPUTE:.*]] = arith.addf %[[EPI_READ]], %[[OUT]]
  // CHECK: vector.transfer_write %[[EPI_COMPUTE]], %[[GLOBAL]][%[[C0]]]

  // 3-stage epilogue: 2 iterations
  // CHECK-3STAGE: gpu.barrier memfence [#gpu.address_space<workgroup>]
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
    // CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
    // CHECK: %[[COMPUTE_READ:.*]] = vector.transfer_read %[[SHARED]][%[[C0]]]
    %2 = vector.transfer_read %alloc[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    // CHECK: %[[COMPUTE:.*]] = arith.addf %[[COMPUTE_READ]], %[[ARG]]
    // CHECK: %[[COMPUTE2:.*]] = arith.addf %[[COMPUTE]], %[[ARG1]]
    %3 = arith.addf %2, %arg2 : vector<1xf32>
    %4 = arith.addf %3, %arg3 : vector<1xf32>
    // CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
    // CHECK: amdgpu.sched_barrier allow = <none>
    // CHECK: vector.transfer_write %[[KER_READ]], %[[SHARED]]
    // CHECK: scf.yield %[[COMPUTE]], %[[COMPUTE2]]
    scf.yield %3, %4 : vector<1xf32>, vector<1xf32>
  }
  // CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
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
    // CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
    // CHECK: %[[COMPUTE_READ:.*]] = vector.transfer_read %[[SHARED]][%[[C0]]]
    %2 = vector.transfer_read %alloc[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    // CHECK: %[[COMPUTE:.*]] = arith.addf %[[COMPUTE_READ]], %[[ARG]]
    %3 = arith.addf %2, %arg2 : vector<1xf32>
    // CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
    // CHECK: amdgpu.sched_barrier allow = <none>
    // CHECK: vector.transfer_write %[[KER_READ]], %[[SHARED]]
    // CHECK: scf.yield %[[COMPUTE]]
    scf.yield %3 : vector<1xf32>
  }
  // CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
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
// CHECK-NOT: gpu.barrier memfence [#gpu.address_space<workgroup>]

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
// CHECK:   gpu.barrier memfence [#gpu.address_space<workgroup>]
// CHECK:   %[[COMPUTE_READ:.*]] = vector.transfer_read %[[WG_ALLOC]][%[[C0]]]
// CHECK:   %[[COMPUTE:.*]] = arith.addf %[[COMPUTE_READ]], %[[ARG]]
// CHECK:   gpu.barrier memfence [#gpu.address_space<workgroup>]
// CHECK:   amdgpu.sched_barrier allow = <none>
// CHECK:   scf.if %[[COND]] {
// CHECK:     %[[COMPUTE_RELOAD:.*]] = vector.transfer_read %[[PRIV_ALLOC2]][%[[C0]]]
// CHECK:     vector.transfer_write %[[COMPUTE_RELOAD]], %[[WG_ALLOC]][%[[C0]]]
// CHECK:   }
// CHECK: scf.yield %[[COMPUTE]]

// CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
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

// CHECK-NOT: gpu.barrier memfence [#gpu.address_space<workgroup>]

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
      gpu.barrier memfence [#gpu.address_space<workgroup>]
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

// CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
// CHECK-NOT: gpu.barrier memfence [#gpu.address_space<workgroup>]

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
// CHECK:   gpu.barrier memfence [#gpu.address_space<workgroup>]
// CHECK:   %[[READ3:.*]] = vector.transfer_read %[[WG_ALLOC]]
// CHECK:   %[[COMP:.*]] = arith.addf
// CHECK:   %[[PRIV_ALLOC3:.*]] = memref.alloca() : memref<1xf32, #gpu.address_space<private>>
// CHECK:   scf.if
// CHECK:   %[[READ4:.*]] = vector.transfer_read %[[PRIV_ALLOC3]]
// CHECK:   gpu.barrier memfence [#gpu.address_space<workgroup>]
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
    // CHECK: gpu.barrier memfence [#gpu.address_space<workgroup>]
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

// CHECK-LABEL: @prefetch_gather_to_lds_two_operands
// CHECK-3STAGE-LABEL: @prefetch_gather_to_lds_two_operands
func.func @prefetch_gather_to_lds_two_operands(
    %A_global: memref<128x128xf32>,
    %B_global: memref<128x128xf32>,
    %C_global: memref<128xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index

  // 2-stage: double-buffered, 3-stage: triple-buffered
  // CHECK: memref.alloc() : memref<2x1xf32, #gpu.address_space<workgroup>>
  // CHECK: memref.alloc() : memref<2x1xf32, #gpu.address_space<workgroup>>
  // CHECK-3STAGE: memref.alloc() : memref<3x1xf32, #gpu.address_space<workgroup>>
  // CHECK-3STAGE: memref.alloc() : memref<3x1xf32, #gpu.address_space<workgroup>>
  %A_lds = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  %B_lds = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>

  // 2-stage: 1 prologue iteration
  // CHECK-COUNT-2: amdgpu.gather_to_lds
  // 3-stage: 2 prologue iterations (N-1 for N stages)
  // CHECK-3STAGE-COUNT-4: amdgpu.gather_to_lds
  // CHECK: scf.for
  // CHECK-3STAGE: scf.for
  %result = scf.for %k = %c0 to %c128 step %c1 iter_args(%acc = %cst) -> (vector<1xf32>) {
    // CHECK: gpu.barrier
    // CHECK-3STAGE: gpu.barrier
    // CHECK-COUNT-2: amdgpu.gather_to_lds
    // CHECK-3STAGE-COUNT-2: amdgpu.gather_to_lds
    amdgpu.gather_to_lds %A_global[%c0, %k], %A_lds[%c0] : vector<1xf32>, memref<128x128xf32>, memref<1xf32, #gpu.address_space<workgroup>>
    amdgpu.gather_to_lds %B_global[%k, %c0], %B_lds[%c0] : vector<1xf32>, memref<128x128xf32>, memref<1xf32, #gpu.address_space<workgroup>>

    // CHECK-COUNT-2: vector.transfer_read
    // CHECK-3STAGE-COUNT-2: vector.transfer_read
    %a_val = vector.transfer_read %A_lds[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    %b_val = vector.transfer_read %B_lds[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    // CHECK: arith.mulf
    // CHECK-3STAGE: arith.mulf
    %prod = arith.mulf %a_val, %b_val : vector<1xf32>
    // CHECK: arith.addf
    // CHECK-3STAGE: arith.addf
    %sum = arith.addf %prod, %acc : vector<1xf32>

    // CHECK: scf.yield
    // CHECK-3STAGE: scf.yield
    scf.yield %sum : vector<1xf32>
  }
  // CHECK: gpu.barrier
  // CHECK: vector.transfer_read
  // CHECK: arith.mulf
  // CHECK-3STAGE: gpu.barrier
  // CHECK-3STAGE: vector.transfer_read
  // CHECK-3STAGE: arith.mulf
  // CHECK-3STAGE: vector.transfer_read
  // CHECK-3STAGE: arith.mulf

  vector.transfer_write %result, %C_global[%c0] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
  return
}

// -----

// CHECK-LABEL: @gather_to_lds_inside_if_not_multibuffered
func.func @gather_to_lds_inside_if_not_multibuffered(
    %global: memref<128x128xf32>,
    %output: memref<128xf32>,
    %bound: index) {
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index

  // CHECK: memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  // CHECK-NOT: memref<2x1xf32
  %lds = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>

  %result = scf.for %k = %c0 to %c128 step %c1 iter_args(%acc = %cst) -> (vector<1xf32>) {
    // CHECK: scf.if
    %in_bounds = arith.cmpi slt, %k, %bound : index
    // CHECK: amdgpu.gather_to_lds
    scf.if %in_bounds {
      amdgpu.gather_to_lds %global[%c0, %k], %lds[%c0] : vector<1xf32>, memref<128x128xf32>, memref<1xf32, #gpu.address_space<workgroup>>
    }
    // CHECK: vector.transfer_read
    %val = vector.transfer_read %lds[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    // CHECK: arith.addf
    %sum = arith.addf %val, %acc : vector<1xf32>
    // CHECK: scf.yield
    scf.yield %sum : vector<1xf32>
  }

  vector.transfer_write %result, %output[%c0] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
  return
}

// -----

// CHECK-ALL-LABEL: @gather_to_lds_mixed_with_stream_copy
// CHECK-ALL-SAME: (%[[GLOBAL:.*]]: memref<128x128xf32>, %[[OUTPUT:.*]]: memref<128xf32>)
func.func @gather_to_lds_mixed_with_stream_copy(
    %global: memref<128x128xf32>,
    %output: memref<128xf32>) {
  // CHECK-ALL-NEXT: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  // CHECK-ALL-NEXT: %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  // CHECK-ALL-NEXT: %[[C128:.*]] = arith.constant 128 : index
  %c128 = arith.constant 128 : index
  // CHECK-ALL-NEXT: %[[C1:.*]] = arith.constant 1 : index
  %c1 = arith.constant 1 : index
  // CHECK-ALL-NEXT: %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index

  // CHECK-ALL-NEXT: %[[LDS_ASYNC:.*]] = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  %lds_async = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  // CHECK-ALL-NEXT: %[[LDS_STREAM:.*]] = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  %lds_stream = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>

  // CHECK-ALL-NEXT: %[[RESULT:.*]] = scf.for %[[K:.*]] = %[[C0]] to %[[C128]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[CST]]) -> (vector<1xf32>) {
  %result = scf.for %k = %c0 to %c128 step %c1 iter_args(%acc = %cst) -> (vector<1xf32>) {
    // CHECK-ALL-NEXT:   amdgpu.gather_to_lds %[[GLOBAL]][%[[C0]], %[[K]]], %[[LDS_ASYNC]][%[[C0]]] : vector<1xf32>, memref<128x128xf32>, memref<1xf32, #gpu.address_space<workgroup>>
    amdgpu.gather_to_lds %global[%c0, %k], %lds_async[%c0] : vector<1xf32>, memref<128x128xf32>, memref<1xf32, #gpu.address_space<workgroup>>

    // CHECK-ALL-NEXT:   %[[VAL_GLOBAL:.*]] = vector.transfer_read %[[GLOBAL]][%[[K]], %[[C0]]], %[[CST_0]]{{.*}} : memref<128x128xf32>, vector<1xf32>
    %val_global = vector.transfer_read %global[%k, %c0], %cst_0 : memref<128x128xf32>, vector<1xf32>
    // CHECK-ALL-NEXT:   vector.transfer_write %[[VAL_GLOBAL]], %[[LDS_STREAM]][%[[C0]]]{{.*}} : vector<1xf32>, memref<1xf32, #gpu.address_space<workgroup>>
    vector.transfer_write %val_global, %lds_stream[%c0] : vector<1xf32>, memref<1xf32, #gpu.address_space<workgroup>>

    // CHECK-ALL-NEXT:   %[[VAL1:.*]] = vector.transfer_read %[[LDS_ASYNC]][%[[C0]]], %[[CST_0]]{{.*}} : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    %val1 = vector.transfer_read %lds_async[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    // CHECK-ALL-NEXT:   %[[VAL2:.*]] = vector.transfer_read %[[LDS_STREAM]][%[[C0]]], %[[CST_0]]{{.*}} : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    %val2 = vector.transfer_read %lds_stream[%c0], %cst_0 : memref<1xf32, #gpu.address_space<workgroup>>, vector<1xf32>
    // CHECK-ALL-NEXT:   %[[SUM1:.*]] = arith.addf %[[VAL1]], %[[ACC]] : vector<1xf32>
    %sum1 = arith.addf %val1, %acc : vector<1xf32>
    // CHECK-ALL-NEXT:   %[[SUM2:.*]] = arith.addf %[[VAL2]], %[[SUM1]] : vector<1xf32>
    %sum2 = arith.addf %val2, %sum1 : vector<1xf32>

    // CHECK-ALL-NEXT:   scf.yield %[[SUM2]] : vector<1xf32>
    scf.yield %sum2 : vector<1xf32>
  }

  // CHECK-ALL-NEXT: }
  // CHECK-ALL-NEXT: vector.transfer_write %[[RESULT]], %[[OUTPUT]][%[[C0]]]{{.*}} : vector<1xf32>, memref<128xf32>
  vector.transfer_write %result, %output[%c0] {in_bounds = [true]} : vector<1xf32>, memref<128xf32>
  // CHECK-ALL-NEXT: return
  return
}

// -----

// CHECK-LABEL: @gather_to_lds_subview_escape_no_multibuffer
// CHECK: memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
// CHECK-NOT: memref<2x1xf32
func.func @gather_to_lds_subview_escape_no_multibuffer(
    %global: memref<128x128xf32>,
    %output: memref<128xf32>) {
  %cst = arith.constant dense<0.000000e+00> : vector<1xf32>
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index

  %lds = memref.alloc() : memref<1xf32, #gpu.address_space<workgroup>>
  %sv = memref.subview %lds[%c0] [1] [1]
        : memref<1xf32, #gpu.address_space<workgroup>>
        to memref<1xf32, strided<[1], offset: ?>, #gpu.address_space<workgroup>>

  // Use the subview outside the loop to force an escape.
  %outside = vector.transfer_read %sv[%c0], %cst_0
             : memref<1xf32, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf32>

  %result = scf.for %k = %c0 to %c128 step %c1 iter_args(%acc = %cst) -> (vector<1xf32>) {
    amdgpu.gather_to_lds %global[%c0, %k], %sv[%c0]
        : vector<1xf32>, memref<128x128xf32>,
          memref<1xf32, strided<[1], offset: ?>, #gpu.address_space<workgroup>>
    %val = vector.transfer_read %sv[%c0], %cst_0
           : memref<1xf32, strided<[1], offset: ?>, #gpu.address_space<workgroup>>, vector<1xf32>
    %sum = arith.addf %val, %acc : vector<1xf32>
    scf.yield %sum : vector<1xf32>
  }

  %combined = arith.addf %result, %outside : vector<1xf32>
  vector.transfer_write %combined, %output[%c0] {in_bounds = [true]}
      : vector<1xf32>, memref<128xf32>
  return
}

// -----

// Test that amdgpu.transpose_load from shared memory is treated as a shared
// memory read for barrier placement.

// CHECK-LABEL: @prefetch_transpose_load
// CHECK-SAME: (%[[GLOBAL:.*]]: memref<128xf16>)
func.func @prefetch_transpose_load(%arg0: memref<128xf16>) {
  %cst = arith.constant dense<0.000000e+00> : vector<4xf16>
  %cst_0 = arith.constant 0.000000e+00 : f16
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<4xf16, #gpu.address_space<workgroup>>
  // CHECK: %[[SHARED:.*]] = memref.alloc() : memref<4xf16, #gpu.address_space<workgroup>>
  // CHECK: vector.transfer_read %[[GLOBAL]]
  // CHECK: vector.transfer_write {{.*}}, %[[SHARED]]
  // CHECK: scf.for
  %0 = scf.for %arg1 = %c0 to %c128 step %c1 iter_args(%arg2 = %cst) -> (vector<4xf16>) {
    %1 = vector.transfer_read %arg0[%arg1], %cst_0 : memref<128xf16>, vector<4xf16>
    vector.transfer_write %1, %alloc[%c0] {in_bounds = [true]} : vector<4xf16>, memref<4xf16, #gpu.address_space<workgroup>>
    // CHECK:      vector.transfer_read %[[GLOBAL]]
    // CHECK:      gpu.barrier memfence [#gpu.address_space<workgroup>]
    // CHECK-NEXT: amdgpu.transpose_load %[[SHARED]]
    // CHECK:      arith.addf
    // CHECK:      gpu.barrier memfence [#gpu.address_space<workgroup>]
    // CHECK-NEXT: amdgpu.sched_barrier allow = <none>
    // CHECK:      vector.transfer_write {{.*}}, %[[SHARED]]
    // CHECK:      scf.yield
    %2 = amdgpu.transpose_load %alloc[%c0] : memref<4xf16, #gpu.address_space<workgroup>> -> vector<4xf16>
    %3 = arith.addf %2, %arg2 : vector<4xf16>
    scf.yield %3 : vector<4xf16>
  }
  // CHECK:      gpu.barrier memfence [#gpu.address_space<workgroup>]
  // CHECK-NEXT: amdgpu.transpose_load %[[SHARED]]
  // CHECK:      arith.addf
  // CHECK:      vector.transfer_write {{.*}}, %[[GLOBAL]]
  vector.transfer_write %0, %arg0[%c0] {in_bounds = [true]} : vector<4xf16>, memref<128xf16>
  return
}
