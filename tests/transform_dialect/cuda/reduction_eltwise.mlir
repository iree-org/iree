!in_tensor_t = tensor<8x64xf32>
!out_tensor_t = tensor<8xf32>

func.func @reduce(%arg : !in_tensor_t) -> (!out_tensor_t) {
  %cst = arith.constant -0.000000e+00 : f32

  %0 = tensor.empty() : !out_tensor_t
  %1 = linalg.fill ins(%cst : f32) outs(%0 : !out_tensor_t) -> !out_tensor_t
  %5 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]}
    ins(%arg : !in_tensor_t) outs(%1 : !out_tensor_t) {
      ^bb0(%arg3: f32, %arg4: f32):
        %4 = arith.addf %arg3, %arg4 : f32
        linalg.yield %4 : f32
      } -> !out_tensor_t

  %6 = tensor.empty() : !out_tensor_t
  %7 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%5 : !out_tensor_t) outs(%6 : !out_tensor_t) {
    ^bb0(%arg3: f32, %arg4: f32):
      %4 = math.sqrt %arg3 : f32
      linalg.yield %4 : f32
    } -> !out_tensor_t
  return %7 : !out_tensor_t
}

/// Note: the current --iree-codegen-llvmgpu-enable-transform-dialect-jit only works for exactly this reduction atm.
// RUN: iree-compile %s --iree-hal-target-backends=cuda | \
// RUN: iree-run-module --module=- --function=reduce --device=cuda --input="8x64xf32=1" |\
// RUN: FileCheck %s --check-prefix=EXEC

  //     CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  //     CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  //     CHECK-DAG: %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
  //     CHECK-DAG: %[[SHMEM_ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x2xf32, #gpu.address_space<workgroup>>
  //     CHECK-DAG: %[[TIDX:.]] = gpu.thread_id  x
  //     CHECK-DAG: %[[TIDY:.]] = gpu.thread_id  y
  //     CHECK-DAG: %[[CONDXIS0:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index

  // Distributed reduction: everyone loads then 5 xor + addf expected
  //         CHECK: vector.transfer_read %{{.*}}[%[[workgroup_id_x]], %[[TIDY]], %[[TIDX]]]
  // CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

  //         CHECK: %[[RES:.*]] = arith.addf %{{.*}}

  //         CHECK: %[[RES_VEC:.*]] = vector.broadcast %[[RES]] : f32 to vector<f32>
  //         CHECK: scf.if %[[CONDXIS0]]
  //         CHECK:   vector.transfer_write %[[RES_VEC]], %[[SHMEM_ALLOC]][%[[C0]], %[[TIDY]]]
  //         CHECK: gpu.barrier

  // Last part is not distributed atm and is only ran by threadIdx.x == 0 and threadIdx.y == 0.
  // It should contain the fused elementwise operation.
  //         CHECK: %[[CONDYIS0:.*]] = arith.cmpi ult, %[[TIDY]], %[[C1]] : index
  //          TODO: cond eq 0 and cond ult 1 do not CSE atm.
  //         CHECK: %[[CONXANDYARE0:.*]] = arith.andi %{{.*}}, %[[CONDYIS0]] : i1
  //         CHECK: scf.if %[[CONXANDYARE0]] {
  //         CHECK:   vector.transfer_read
  //         CHECK:   vector.reduction <add>
  //         CHECK:   math.sqrt
  //         CHECK:   vector.transfer_write
  //         CHECK: gpu.barrier

//      EXEC: result[0]: hal.buffer_view
// EXEC-NEXT: 8xf32=8 8 8 8 8 8 8 8
