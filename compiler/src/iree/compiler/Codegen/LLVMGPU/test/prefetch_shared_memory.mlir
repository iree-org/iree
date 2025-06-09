// RUN: iree-opt -pass-pipeline="builtin.module(func.func(iree-llvmgpu-prefetch-shared-memory),cse,canonicalize)" %s --split-input-file | FileCheck %s

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
    // CHECK-DAG: %[[IVPLUS1:.*]] = arith.addi %[[IV]], %[[C1]]  overflow<nsw> : index
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

// -----

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

// -----

// CHECK-LABEL: @prefetch_add_with_if
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

// CHECK-LABEL: @noprefetch_copyback
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

// CHECK-LABEL: @prefetch_scf_if
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
// CHECK:   %[[IVP1:.*]] = arith.addi %[[IV]], %[[C1]]
// CHECK:   %[[PRIV_ALLOC2:.*]] = memref.alloca() : memref<1xf32, #gpu.address_space<private>>
// CHECK:   scf.if %[[COND]] {
// CHECK:     %[[KER_READ:.*]] = vector.transfer_read %[[GLOBAL]][%[[IVP1]]]
// CHECK:     vector.transfer_write %[[KER_READ]], %[[PRIV_ALLOC2]][%[[C0]]]
// CHECK:   }
// CHECK:   gpu.barrier
// CHECK:   %[[COMPUTE_READ:.*]] = vector.transfer_read %[[WG_ALLOC]][%[[C0]]]
// CHECK:   %[[COMPUTE:.*]] = arith.addf %[[COMPUTE_READ]], %[[ARG]]
// CHECK:   gpu.barrier
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

// CHECK-LABEL: @noprefetch_scf_if_readwritetogether
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

// CHECK-LABEL: @noprefetch_unsupportedif
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

// CHECK-LABEL: @prefetch_scf_if_transientreadwrite
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
// CHECK: scf.if
// CHECK-DAG: %[[PRIV_ALLOC1:.*]] = memref.alloca() : memref<1xf32, #gpu.address_space<private>>
// CHECK: scf.if
// CHECK: %[[INIT:.*]] = vector.transfer_read %[[PRIV_ALLOC1]]
// CHECK: vector.transfer_write %[[INIT]], %[[WG_ALLOC]]
// CHECK: %[[OUT:.*]] = scf.for
// CHECK:   %[[PRIV_ALLOC2:.*]] = memref.alloca() : memref<1xf32, #gpu.address_space<private>>
// CHECK:   scf.if
// CHECK:   gpu.barrier
// CHECK:   %[[READ3:.*]] = vector.transfer_read %[[WG_ALLOC]]
// CHECK:   %[[COMP:.*]] = arith.addf
// CHECK:   gpu.barrier
// CHECK:   %[[PRIV_ALLOC3:.*]] = memref.alloca() : memref<1xf32, #gpu.address_space<private>>
// CHECK:   scf.if
// CHECK:   %[[READ5:.*]] = vector.transfer_read %[[PRIV_ALLOC3]]
// CHECK:   vector.transfer_write %[[READ5]], %[[WG_ALLOC]]
// CHECK:   scf.yield %[[COMP]]
