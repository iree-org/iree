!in_tensor_t = tensor<8x64xf32>
!out_tensor_t = tensor<8xf32>

func.func @reduce(%arg : !in_tensor_t) -> (!out_tensor_t) {
  %cst = arith.constant -0.000000e+00 : f32

  %0 = tensor.empty() : !out_tensor_t
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_t) ->  !out_tensor_t
  %2 = tensor.empty() : !in_tensor_t
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%arg : !in_tensor_t) outs(%2 : !in_tensor_t) {
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = arith.addf %arg3, %arg3 : f32
      %5 = arith.addf %4, %4 : f32
      linalg.yield %5 : f32
    } -> !in_tensor_t

  %6 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%3 : !in_tensor_t) outs(%1 : !out_tensor_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> !out_tensor_t

  return %6 : !out_tensor_t
}

// RUN: iree-opt %s --iree-hal-target-backends=cuda \
// RUN:     --iree-abi-transformation-pipeline \
// RUN:     --iree-flow-transformation-pipeline \
// RUN:     --iree-stream-transformation-pipeline \
// RUN:     --iree-hal-configuration-pipeline | \
// RUN: FileCheck %s --check-prefix=DISPATCH

// RUN: iree-opt %s --iree-hal-target-backends=cuda \
// RUN:     --iree-abi-transformation-pipeline \
// RUN:     --iree-flow-transformation-pipeline \
// RUN:     --iree-stream-transformation-pipeline \
// RUN:     --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))'
// RUN:     --iree-codegen-llvmgpu-use-transform-dialect=%p/%S_codegen_spec.mlir | \
// RUN: FileCheck %s

// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit | \
// RUN: iree-run-module --module=- --function=reduce --device=cuda --input="8x64xf32=1" |\
// RUN: FileCheck %s --check-prefix=EXEC

// Check that both generics ended up in the same region.
// DISPATCH:     hal.executable.variant
// DISPATCH:     linalg.fill
// DISPATCH-NOT: hal.executable.variant
// DISPATCH:     linalg.generic
// DISPATCH-NOT: hal.executable.variant
// DISPATCH:     linalg.generic

//     CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//     CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
//     CHECK-DAG: %[[F0:.*]] = arith.constant dense<0.000000e+00> : vector<f32>
//     CHECK-DAG: %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
//     CHECK-DAG: %[[SHMEM_ALLOC:.*]] = memref.alloc() {alignment = 128 : i64} : memref<1x2xf32, 3>
//     CHECK-DAG: %[[TIDX:.]] = gpu.thread_id  x
//     CHECK-DAG: %[[TIDY:.]] = gpu.thread_id  y
//     CHECK-DAG: %[[TIDZ:.]] = gpu.thread_id  z

//         CHECK: %[[SHMEM_VIEW_EXPANDED:.*]] = memref.subview %[[SHMEM_ALLOC]][%[[TIDZ]], %[[TIDY]]]{{.*}}to memref<f32, {{.*}}, 3>

// Distributed reduction: everyone loads, does the elementwise then 5 xor + addf expected
//         CHECK: vector.transfer_read %{{.*}}[%[[TIDZ]], %[[TIDY]], %[[TIDX]]]
//         CHECK: arith.addf
//         CHECK: arith.addf
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

//         CHECK: %[[RES:.*]] = arith.addf %{{.*}}

//         CHECK: %[[RES_VEC:.*]] = vector.broadcast %[[RES]] : f32 to vector<f32>
//         CHECK: %[[CONDXIS0:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index
//         CHECK: scf.if %[[CONDXIS0]]
//         CHECK:   vector.transfer_write %[[RES_VEC]], %[[SHMEM_VIEW_EXPANDED]][]
//         CHECK: gpu.barrier

// Last part is not distributed atm and is only ran by threadIdx.x == 0 and threadIdx.y == 0.
//         CHECK: %[[CONDYIS0:.*]] = arith.cmpi ult, %[[TIDY]], %[[C1]] : index
//          TODO: cond eq 0 and cond ult 1 do not CSE atm.
//         CHECK: %[[CONXANDYARE0:.*]] = arith.andi %{{.*}}, %[[CONDYIS0]] : i1
//         CHECK: scf.if %[[CONXANDYARE0]] {
//         CHECK:   vector.transfer_read
//         CHECK:   vector.reduction <add>
//         CHECK:   vector.transfer_write
//         CHECK: gpu.barrier
//         CHECK: memref.dealloc %[[SHMEM_ALLOC]] : memref<1x2xf32, 3>

//      EXEC: result[0]: hal.buffer_view
// EXEC-NEXT: 8xf32=256 256 256 256 256 256 256 256
