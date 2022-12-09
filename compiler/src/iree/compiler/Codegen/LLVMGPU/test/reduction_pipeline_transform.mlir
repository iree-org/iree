// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))" --iree-codegen-llvmgpu-enable-transform-dialect-jit %s | FileCheck %s

hal.executable @group_reduction {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @group_reduction ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<8x64xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x64xf32>> -> tensor<8x64xf32>
      %3 = tensor.empty() : tensor<8xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<8x64xf32>) outs(%4 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<8xf32>
      flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      return
    }
  }
}
}
//   CHECK-LABEL: func.func @group_reduction
//     CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//     CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//     CHECK-DAG:   %[[F0:.*]] = arith.constant dense<0.000000e+00> : vector<f32>
//     CHECK-DAG:   %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
//     CHECK-DAG:   %[[SHMEM_ALLOC:.*]] = memref.alloc() {alignment = 128 : i64} : memref<1x2xf32, 3>
//     CHECK-DAG:   %[[TIDX:.]] = gpu.thread_id  x
//     CHECK-DAG:   %[[TIDY:.]] = gpu.thread_id  y
//     CHECK-DAG:   %[[TIDZ:.]] = gpu.thread_id  z
//         CHECK:   %[[SHMEM_VIEW_EXPANDED:.*]] = memref.subview %[[SHMEM_ALLOC]][%[[TIDZ]], %[[TIDY]]]{{.*}}to memref<f32, {{.*}}, 3>

// Distributed reduction: everyone loads then 5 xor + addf expected
//         CHECK:   vector.transfer_read %{{.*}}[%[[TIDZ]], %[[TIDY]], %[[TIDX]]]
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf
//         CHECK:   %[[RES:.*]] = arith.addf %{{.*}}
//         CHECK:   %[[RES_VEC:.*]] = vector.broadcast %[[RES]] : f32 to vector<f32>
//         CHECK:   %[[CONDXIS0:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index
//         CHECK:   scf.if %[[CONDXIS0]]
//         CHECK:     vector.transfer_write %[[RES_VEC]], %[[SHMEM_VIEW_EXPANDED]][]
//         CHECK:   gpu.barrier

// Last part is not distributed atm and is only ran by threadIdx.x == 0 and threadIdx.y == 0.
//         CHECK:   %[[CONDYIS0:.*]] = arith.cmpi ult, %[[TIDY]], %[[C1]] : index
//          TODO: cond eq 0 and cond ult 1 do not CSE atm.
//         CHECK:   %[[CONXANDYARE0:.*]] = arith.andi %{{.*}}, %[[CONDYIS0]] : i1
//         CHECK:   scf.if %[[CONXANDYARE0]] {
//         CHECK:     vector.transfer_read
//         CHECK:     vector.reduction <add>
//         CHECK:     vector.transfer_write
//         CHECK:   gpu.barrier
//         CHECK:   memref.dealloc %[[SHMEM_ALLOC]] : memref<1x2xf32, 3>

// -----

hal.executable @group_reduction_elementwise {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @group_reduction_elementwise ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction_elementwise() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<8x64xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x64xf32>> -> tensor<8x64xf32>
      %3 = tensor.empty() : tensor<8xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<8x64xf32>) outs(%4 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = arith.addf %in, %out : f32
        linalg.yield %7 : f32
      } -> tensor<8xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%5 : tensor<8xf32>) outs(%3 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = math.sqrt %in : f32
        linalg.yield %7 : f32
      } -> tensor<8xf32>
      flow.dispatch.tensor.store %6, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_reduction_elementwise
//     CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//     CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//     CHECK-DAG:   %[[F0:.*]] = arith.constant dense<0.000000e+00> : vector<f32>
//     CHECK-DAG:   %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
//     CHECK-DAG:   %[[SHMEM_ALLOC:.*]] = memref.alloc() {alignment = 128 : i64} : memref<1x2xf32, 3>
//     CHECK-DAG:   %[[TIDX:.]] = gpu.thread_id  x
//     CHECK-DAG:   %[[TIDY:.]] = gpu.thread_id  y
//     CHECK-DAG:   %[[TIDZ:.]] = gpu.thread_id  z
//         CHECK:   %[[SHMEM_VIEW_EXPANDED:.*]] = memref.subview %[[SHMEM_ALLOC]][%[[TIDZ]], %[[TIDY]]]{{.*}}to memref<f32, {{.*}}, 3>

// Distributed reduction: everyone loads then 5 xor + addf expected
//         CHECK:   vector.transfer_read %{{.*}}[%[[TIDZ]], %[[TIDY]], %[[TIDX]]]
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf
//         CHECK:   %[[RES:.*]] = arith.addf %{{.*}}
//         CHECK:   %[[RES_VEC:.*]] = vector.broadcast %[[RES]] : f32 to vector<f32>
//         CHECK:   %[[CONDXIS0:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index
//         CHECK:   scf.if %[[CONDXIS0]]
//         CHECK:     vector.transfer_write %[[RES_VEC]], %[[SHMEM_VIEW_EXPANDED]][]
//         CHECK:   gpu.barrier

// Last part is not distributed atm and is only ran by threadIdx.x == 0 and threadIdx.y == 0.
// It should contain the fused elementwise operation.
//         CHECK:   %[[CONDYIS0:.*]] = arith.cmpi ult, %[[TIDY]], %[[C1]] : index
//          TODO:   cond eq 0 and cond ult 1 do not CSE atm.
//         CHECK:   %[[CONXANDYARE0:.*]] = arith.andi %{{.*}}, %[[CONDYIS0]] : i1
//         CHECK:   scf.if %[[CONXANDYARE0]] {
//         CHECK:     vector.transfer_read
//         CHECK:     vector.reduction <add>
//         CHECK:     math.sqrt
//         CHECK:     vector.transfer_write
//         CHECK:   gpu.barrier
//         CHECK:   memref.dealloc %[[SHMEM_ALLOC]] : memref<1x2xf32, 3>

// -----

hal.executable @group_elementwise_reduction {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @group_elementwise_reduction ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_elementwise_reduction() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<8x64xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x64xf32>> -> tensor<8x64xf32>
      %3 = tensor.empty() : tensor<8xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<8x64xf32>) outs(%4 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %in : f32
        %7 = arith.addf %6, %6 : f32
        %8 = arith.addf %7, %out : f32
        linalg.yield %8 : f32
      } -> tensor<8xf32>
      flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_elementwise_reduction
//     CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//     CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//     CHECK-DAG:   %[[F0:.*]] = arith.constant dense<0.000000e+00> : vector<f32>
//     CHECK-DAG:   %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
//     CHECK-DAG:   %[[SHMEM_ALLOC:.*]] = memref.alloc() {alignment = 128 : i64} : memref<1x2xf32, 3>
//     CHECK-DAG:   %[[TIDX:.]] = gpu.thread_id  x
//     CHECK-DAG:   %[[TIDY:.]] = gpu.thread_id  y
//     CHECK-DAG:   %[[TIDZ:.]] = gpu.thread_id  z

//         CHECK:   %[[SHMEM_VIEW_EXPANDED:.*]] = memref.subview %[[SHMEM_ALLOC]][%[[TIDZ]], %[[TIDY]]]{{.*}}to memref<f32, {{.*}}, 3>

// Distributed reduction: everyone loads, does the elementwise then 5 xor + addf expected
//         CHECK:   vector.transfer_read %{{.*}}[%[[TIDZ]], %[[TIDY]], %[[TIDX]]]
//         CHECK:   arith.addf
//         CHECK:   arith.addf
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

//         CHECK:   %[[RES:.*]] = arith.addf %{{.*}}

//         CHECK:   %[[RES_VEC:.*]] = vector.broadcast %[[RES]] : f32 to vector<f32>
//         CHECK:   %[[CONDXIS0:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index
//         CHECK:   scf.if %[[CONDXIS0]]
//         CHECK:     vector.transfer_write %[[RES_VEC]], %[[SHMEM_VIEW_EXPANDED]][]
//         CHECK:   gpu.barrier

// Last part is not distributed atm and is only ran by threadIdx.x == 0 and threadIdx.y == 0.
//         CHECK:   %[[CONDYIS0:.*]] = arith.cmpi ult, %[[TIDY]], %[[C1]] : index
//          TODO: cond eq 0 and cond ult 1 do not CSE atm.
//         CHECK:   %[[CONXANDYARE0:.*]] = arith.andi %{{.*}}, %[[CONDYIS0]] : i1
//         CHECK:   scf.if %[[CONXANDYARE0]] {
//         CHECK:     vector.transfer_read
//         CHECK:     vector.reduction <add>
//         CHECK:     vector.transfer_write
//         CHECK:   gpu.barrier
//         CHECK:   memref.dealloc %[[SHMEM_ALLOC]] : memref<1x2xf32, 3>

// -----

hal.executable @group_elementwise_reduction_elementwise {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @group_elementwise_reduction_elementwise ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_elementwise_reduction_elementwise() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:tensor<8x64xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x64xf32>> -> tensor<8x64xf32>
      %3 = tensor.empty() : tensor<8xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<8xf32>) -> tensor<8xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<8x64xf32>) outs(%4 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = arith.addf %in, %in : f32
        %8 = arith.addf %7, %7 : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      } -> tensor<8xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%5 : tensor<8xf32>) outs(%3 : tensor<8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = math.sqrt %in : f32
        linalg.yield %7 : f32
      } -> tensor<8xf32>
      flow.dispatch.tensor.store %6, %1, offsets = [0], sizes = [8], strides = [1] : tensor<8xf32> -> !flow.dispatch.tensor<writeonly:tensor<8xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_elementwise_reduction_elementwise
//     CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//     CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//     CHECK-DAG:   %[[F0:.*]] = arith.constant dense<0.000000e+00> : vector<f32>
//     CHECK-DAG:   %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
//     CHECK-DAG:   %[[SHMEM_ALLOC:.*]] = memref.alloc() {alignment = 128 : i64} : memref<1x2xf32, 3>
//     CHECK-DAG:   %[[TIDX:.]] = gpu.thread_id  x
//     CHECK-DAG:   %[[TIDY:.]] = gpu.thread_id  y
//     CHECK-DAG:   %[[TIDZ:.]] = gpu.thread_id  z

//         CHECK:   %[[SHMEM_VIEW_EXPANDED:.*]] = memref.subview %[[SHMEM_ALLOC]][%[[TIDZ]], %[[TIDY]]]{{.*}}to memref<f32, {{.*}}, 3>

// Distributed reduction: everyone loads, does the elementwise then 5 xor + addf expected
//         CHECK:   vector.transfer_read %{{.*}}[%[[TIDZ]], %[[TIDY]], %[[TIDX]]]
//         CHECK:   arith.addf
//         CHECK:   arith.addf
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

//         CHECK:   %[[RES:.*]] = arith.addf %{{.*}}

//         CHECK:   %[[RES_VEC:.*]] = vector.broadcast %[[RES]] : f32 to vector<f32>
//         CHECK:   %[[CONDXIS0:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index
//         CHECK:   scf.if %[[CONDXIS0]]
//         CHECK:     vector.transfer_write %[[RES_VEC]], %[[SHMEM_VIEW_EXPANDED]][]
//         CHECK:   gpu.barrier

// Last part is not distributed atm and is only ran by threadIdx.x == 0 and threadIdx.y == 0.
//         CHECK:   %[[CONDYIS0:.*]] = arith.cmpi ult, %[[TIDY]], %[[C1]] : index
//          TODO: cond eq 0 and cond ult 1 do not CSE atm.
//         CHECK:   %[[CONXANDYARE0:.*]] = arith.andi %{{.*}}, %[[CONDYIS0]] : i1
//         CHECK:   scf.if %[[CONXANDYARE0]] {
//         CHECK:     vector.transfer_read
//         CHECK:     vector.reduction <add>
//         CHECK:     math.sqrt
//         CHECK:     vector.transfer_write
//         CHECK:   gpu.barrier
//         CHECK:   memref.dealloc %[[SHMEM_ALLOC]] : memref<1x2xf32, 3>
