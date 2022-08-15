// RUN: iree-opt %s -iree-hal-target-backends=cuda \
// RUN:     -iree-abi-transformation-pipeline \
// RUN:     -iree-flow-transformation-pipeline  \
// RUN:     -iree-flow-dispatch-use-transform-dialect=%p/reduction_dispatch_spec.mlir \
// RUN:     -iree-stream-transformation-pipeline \
// RUN:     -iree-hal-configuration-pipeline | \
// RUN: iree-opt -pass-pipeline='hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target-pass))' \
// RUN:     -iree-codegen-llvmgpu-use-transform-dialect=%p/reduction_codegen_spec.mlir | \
// RUN: FileCheck %s

// RUN: iree-compile %s -iree-hal-target-backends=cuda \
// RUN:     -iree-flow-dispatch-use-transform-dialect=%p/reduction_dispatch_spec.mlir \
// RUN:     -iree-codegen-llvmgpu-use-transform-dialect=%p/reduction_codegen_spec.mlir |\
// RUN: iree-run-module --entry_function=reduce --device=cuda |\
// RUN: FileCheck %s --check-prefix=EXEC

func.func @reduce() -> (tensor<8xf32>) {
  %cst = arith.constant -0.000000e+00 : f32

  // Note: arith.constant is good for our purposes here but it may be useful to use
  // util.unfoldable_constant.
  %arg = arith.constant dense<1.0> : tensor<8x64xf32>
  %0 = linalg.init_tensor [8] : tensor<8xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<8xf32>) ->   tensor<8xf32>

  //     CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  //     CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  //     CHECK-DAG: %[[F0:.*]] = arith.constant dense<0.000000e+00> : vector<1xf32>
  //     CHECK-DAG: %[[TIDX:.]] = gpu.thread_id  x
  //     CHECK-DAG: %[[TIDY:.]] = gpu.thread_id  y
  //     CHECK-DAG: %[[TIDZ:.]] = gpu.thread_id  z
  //     CHECK-DAG: %[[SHMEM_ALLOC:.*]] = memref.alloc() {alignment = 128 : i64} : memref<4x2xf32, 3>
  //         CHECK: %[[SHMEM_VIEW:.*]] = memref.subview %[[SHMEM_ALLOC]][%[[TIDZ]], %[[TIDY]]]
  //         CHECK: %[[SHMEM_VIEW_EXPANDED:.*]] = memref.expand_shape %[[SHMEM_VIEW]] [] : memref<f32, {{.*}}> into memref<1x1xf32, {{.*}}>
  //         CHECK: %[[CONDXIS0:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index

  // Distributed fill to shared memory, only threadIdx.x == 0 writes.
  //         CHECK: scf.if %[[CONDXIS0]]
  //         CHECK:   vector.transfer_write %[[F0]], %[[SHMEM_VIEW_EXPANDED]][%[[C0]], %[[C0]]]
  //         CHECK: gpu.barrier

  // Note some inefficiencies here: all threads read + addf but only threadIdx.x==0 commits.
  // So only threadIdx.x == 0 could do that.
  // Additionally, the value read is exactly the "Distributed fill to shared memory" from above
  // and there is no interleaved read/write so we could fold this read into only
  // %[[F0]] and only write back to shared memory.
  //
  // Note: This will probably happen once the fill is fused into the split op at the linalg level.
  //         CHECK: %[[NEUTRAL_VEC:.*]] = vector.transfer_read %[[SHMEM_ALLOC]][%[[TIDZ]], %[[TIDY]]]{{.*}}vector<1xf32>

  // TODO: Some foldings are missing, no need to be in vector<1xf32>.
  //         CHECK: %[[NEUTRAL:.*]] = vector.extract %[[NEUTRAL_VEC]][0] : vector<1xf32>
  // Distributed reduction: everyone loads then 5 xor + addf expected
  //         CHECK: vector.transfer_read %{{.*}}[%[[C0]], %[[C0]], %[[TIDX]]]
  // CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

  //         CHECK: %[[RES:.*]] = arith.addf %{{.*}}, %[[NEUTRAL]]

  // TODO: Some foldings are missing, no need to be in vector<1xf32>.
  //         CHECK: %[[RES_VEC:.*]] = vector.broadcast %[[RES]] : f32 to vector<1xf32>
  //         CHECK: scf.if %[[CONDXIS0]]
  //         CHECK:   vector.transfer_write %[[RES_VEC]], %[[SHMEM_VIEW_EXPANDED]][%[[C0]], %[[C0]]]
  //         CHECK: gpu.barrier

  // Last part is not distributed atm and is only ran by threadIdx.x == 0 and threadIdx.y == 0.
  //         CHECK: %[[CONDYIS0:.*]] = arith.cmpi ult, %[[TIDY]], %[[C1]] : index
  //          TODO: cond eq 0 and cond ult 1 do not CSE atm.
  //         CHECK: %[[CONXANDYARE0:.*]] = arith.andi %{{.*}}, %[[CONDYIS0]] : i1
  //         CHECK: scf.if %[[CONXANDYARE0]] {
  // CHECK-COUNT-2:   vector.transfer_read
  //         CHECK:   vector.reduction <add>
  //         CHECK:   vector.transfer_write
  //         CHECK: gpu.barrier
  //         CHECK: memref.dealloc %[[SHMEM_ALLOC]] : memref<4x2xf32, 3>
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%arg : tensor<8x64xf32>) outs(%1 : tensor<8xf32>) {
      ^bb0(%arg3: f32, %arg4: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
      } -> tensor<8xf32>
  return %2 : tensor<8xf32>
}

//      EXEC: result[0]: hal.buffer_view
// EXEC-NEXT: 8xf32=64 64 64 64 64 64 64 64
