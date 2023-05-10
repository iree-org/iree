// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))" %s | FileCheck %s

hal.executable @small_reduction {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @small_reduction ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @small_reduction() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1024x13xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1024xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 13], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1024x13xf32>> -> tensor<1024x13xf32>
      %3 = tensor.empty() : tensor<1024xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<1024xf32>) -> tensor<1024xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<1024x13xf32>) outs(%4 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<1024xf32>
      flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [1024], strides = [1] : tensor<1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<1024xf32>>
      return
    }
  }
}
}

// Small reduction computes the whole reduction on a single thread.
//   CHECK-LABEL: func.func @small_reduction
//     CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
//     CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
//     CHECK-DAG: %[[C12:.*]] = arith.constant 12 : index
//     CHECK-NOT:   memref.alloc()
//         CHECK: gpu.thread_id  x
//         CHECK: %[[v:.*]] = scf.for %{{.*}} = %[[C0]] to %[[C12]] step %[[C4]] {{.*}} -> (vector<1xf32>) {
//         CHECK:   vector.transfer_read {{.*}}: memref<1024x13xf32>, vector<1x4xf32>
//         CHECK:   vector.multi_reduction <add>, %{{.*}} : vector<1x4xf32> to vector<1xf32>
//         CHECK: }
//     CHECK-NOT: gpu.barrier
//         CHECK: %[[r:.*]] = vector.transfer_read {{.*}}: memref<1024x13xf32>, vector<1x1xf32>
//         CHECK: %[[r1:.*]] = vector.shape_cast %[[r:.*]] : vector<1x1xf32> to vector<1xf32>
//         CHECK: arith.addf %[[v]], %[[r1]] : vector<1xf32>

// -----

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
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<8x64xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8xf32>>
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
//     CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
//     CHECK-DAG:   %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
//     CHECK-DAG:   %[[TIDX:.]] = gpu.thread_id  x

// No allocation created for the per thread data.
//     CHECK-NOT:   memref.alloc()

// Fusion occurred, no barrier before the loop
//     CHECK-NOT: gpu.barrier
// Local per-thread scf.for-based reduction.
//         CHECK: %[[v:.*]] = scf.for {{.*}} -> (vector<1xf32>) {
//         CHECK:   vector.transfer_read {{.*}} memref<8x64xf32>, vector<1xf32>
//         CHECK:   arith.addf {{.*}} : vector<1xf32>
// No barrier within the loop.
//     CHECK-NOT:   gpu.barrier
//         CHECK:   }
// No store after the loop, the data are kept in register.
//     CHECK-NOT:   vector.transfer_write
// Barrier after the loop.
//         CHECK:   gpu.barrier

//     CHECK-NOT:   vector.transfer_read
// CHECK-COUNT-5:   gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

//         CHECK:   %[[RES:.*]] = arith.addf %{{.*}} : f32
//         CHECK:   %[[RES_VEC:.*]] = vector.broadcast %{{.*}} : f32 to vector<1xf32>
//         CHECK:   %[[CONDXIS0:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index
//         CHECK:   scf.if %[[CONDXIS0]]
//         CHECK:     vector.transfer_write %[[RES_VEC]]
//         CHECK:   gpu.barrier

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
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<8x64xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8xf32>>
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
//     CHECK-DAG:   %[[C32:.*]] = arith.constant 32 : index
//     CHECK-DAG:   %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
//     CHECK-NOT:   memref.alloc()

// Fusion occurred, no barrier before the loop
//     CHECK-NOT: gpu.barrier
// Local per-thread scf.for-based reduction.
//         CHECK: %[[TIDX:.]] = gpu.thread_id  x
//         CHECK: %[[v:.*]] = scf.for {{.*}} -> (vector<1xf32>)
//         CHECK:   vector.transfer_read {{.*}} vector<1xf32>
//         CHECK:   arith.addf{{.*}} : vector<1xf32>
//         CHECK:   arith.addf{{.*}} : vector<1xf32>
//         CHECK:   arith.addf{{.*}} : vector<1xf32>
// No barrier within the loop
//     CHECK-NOT:   gpu.barrier
//         CHECK: }
//     CHECK-NOT: vector.transfer_write
// Barrier after the loop
//         CHECK:   gpu.barrier

//     CHECK-NOT:   vector.transfer_read
// CHECK-COUNT-5:   gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

//         CHECK:   %[[PARTIAL:.*]] = arith.addf %{{.*}}
//         CHECK:   %[[BROADCAST:.*]], %{{.*}} = gpu.shuffle  idx %[[PARTIAL]]
//         CHECK:   %[[RES_VEC:.*]] = vector.broadcast %[[BROADCAST]] : f32 to vector<1xf32>
//         CHECK:   %[[SQRT_VEC:.*]] = math.sqrt %[[RES_VEC]] : vector<1xf32>
//         CHECK:   %[[CONDXIS0:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index
//         CHECK:   scf.if %[[CONDXIS0]]
//         CHECK:     vector.transfer_write %[[SQRT_VEC]], {{.*}} : vector<1xf32>, memref<8xf32>
//         CHECK:   gpu.barrier

// -----

hal.executable @group_reduction_larger {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @group_reduction_larger ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction_larger() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<33x1024xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<33xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [33, 1024], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<33x1024xf32>> -> tensor<33x1024xf32>
      %3 = tensor.empty() : tensor<33xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<33xf32>) -> tensor<33xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<33x1024xf32>) outs(%4 : tensor<33xf32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<33xf32>
      flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [8], strides = [1] : tensor<33xf32> -> !flow.dispatch.tensor<writeonly:tensor<33xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_reduction_larger
//     CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//     CHECK-DAG:   %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
//     CHECK-NOT:   memref.alloc()

// Fusion occurred, no barrier before the loop
//     CHECK-NOT: gpu.barrier
// Local per-thread scf.for-based reduction.
//         CHECK: %[[TIDX:.]] = gpu.thread_id  x
//         CHECK: %[[TIDX_TIMES_4:.]] = affine.apply{{.*}}[%[[TIDX]]]
//         CHECK: scf.for {{.*}} -> (vector<1xf32>) {
//         CHECK:   vector.transfer_read {{.*}} vector<4xf32>
//         CHECK:   vector.reduction <add>{{.*}} : vector<4xf32> into f32
//         CHECK:   vector.broadcast {{.*}} : f32 to vector<1xf32>
// No barrier within the loop
//     CHECK-NOT:   gpu.barrier
//         CHECK: }
//     CHECK-NOT: vector.transfer_write
//     CHECK-NOT: vector.transfer_read
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf
//         CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<8xf32, #gpu.address_space<workgroup>>
//         CHECK: scf.if
//         CHECK:   memref.store %{{.*}}, %[[ALLOC]][%{{.*}}] : memref<8xf32, #gpu.address_space<workgroup>>
//         CHECK: }
//         CHECK: arith.minui
//         CHECK: memref.load
// CHECK-COUNT-3: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf
//         CHECK: %[[RES:.*]], %{{.*}} = gpu.shuffle  idx
//         CHECK: %[[RES_VEC:.*]] = vector.broadcast %[[RES]] : f32 to vector<1xf32>
//         CHECK: %[[CONDXIS0:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index
//         CHECK: scf.if %[[CONDXIS0]]
//         CHECK:   vector.transfer_write %[[RES_VEC]]
//         CHECK: gpu.barrier

// -----

hal.executable @group_reduction_1d {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @group_reduction_1d ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction_1d() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<64xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<f32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [64], strides = [1] : !flow.dispatch.tensor<readonly:tensor<64xf32>> -> tensor<64xf32>
      %3 = tensor.empty() : tensor<f32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<f32>) -> tensor<f32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%2 : tensor<64xf32>) outs(%4 : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %6 = arith.addf %in, %out : f32
        linalg.yield %6 : f32
      } -> tensor<f32>
      flow.dispatch.tensor.store %5, %1, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:tensor<f32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_reduction_1d
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

// -----

hal.executable @group_elementwise_reduction_elementwise_4d {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @group_elementwise_reduction_elementwise_4d ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_elementwise_reduction_elementwise_4d() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant -0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x4x8x64xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x4x8xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 4, 8, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4x8x64xf32>> -> tensor<2x4x8x64xf32>
      %3 = tensor.empty() : tensor<2x4x8xf32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>],
                           iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%2 : tensor<2x4x8x64xf32>) outs(%4 : tensor<2x4x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = arith.addf %in, %in : f32
        %8 = arith.addf %7, %7 : f32
        %9 = arith.addf %8, %out : f32
        linalg.yield %9 : f32
      } -> tensor<2x4x8xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                           iterator_types = ["parallel", "parallel", "parallel"]} ins(%5 : tensor<2x4x8xf32>) outs(%3 : tensor<2x4x8xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = math.sqrt %in : f32
        linalg.yield %7 : f32
      } -> tensor<2x4x8xf32>
      flow.dispatch.tensor.store %6, %1, offsets = [0, 0, 0], sizes = [2, 4, 8], strides = [1, 1, 1] : tensor<2x4x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x4x8xf32>>
      return
    }
  }
}
}

//   CHECK-LABEL: func.func @group_elementwise_reduction_elementwise_4d
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}{{.*}} arith.addf

// -----

hal.executable @group_reduction_i8_12345 {
hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
  hal.executable.export public @group_reduction_i8_12345 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @group_reduction_i8_12345() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0 : i8
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<8x12345xi8>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x12345xi8>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 12345], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x12345xi8>> -> tensor<8x12345xi8>
      %3 = tensor.empty() : tensor<8x12345xi8>
      %4 = tensor.empty() : tensor<8xi8>
      %5 = linalg.fill ins(%cst : i8) outs(%4 : tensor<8xi8>) -> tensor<8xi8>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
                           iterator_types = ["parallel", "reduction"]}
        ins(%2 : tensor<8x12345xi8>)
       outs(%5 : tensor<8xi8>) {
      ^bb0(%in: i8, %out: i8):
        %6 = arith.addi %in, %out : i8
        linalg.yield %6 : i8
      } -> tensor<8xi8>
      %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
                           iterator_types = ["parallel", "parallel"]}
        ins(%2, %6 : tensor<8x12345xi8>, tensor<8xi8>)
       outs(%3 : tensor<8x12345xi8>) {
      ^bb0(%in: i8, %in_0: i8, %out: i8):
        %8 = arith.divui %in, %in_0 : i8
        linalg.yield %8 : i8
      } -> tensor<8x12345xi8>
      flow.dispatch.tensor.store %7, %1, offsets = [0, 0], sizes = [8, 12345], strides = [1, 1] : tensor<8x12345xi8> -> !flow.dispatch.tensor<writeonly:tensor<8x12345xi8>>
      return
    }
  }
}
}


//   CHECK-LABEL: func.func @group_reduction_i8_12345
//     CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//     CHECK-DAG:   %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index

//         CHECK: %[[ALLOC0:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1xi8, #gpu.address_space<workgroup>>
// Local per-thread scf.for-based reduction.
//         CHECK: %[[TIDX:.]] = gpu.thread_id  x
//         CHECK: scf.for {{.*}} -> (vector<1xi8>)
//         CHECK:   vector.transfer_read {{.*}} vector<1xi8>
//         CHECK:   arith.addi{{.*}} : vector<1xi8>
//     CHECK-NOT:   vector.transfer_write
// No barrier within the loop
//     CHECK-NOT:   gpu.barrier
//         CHECK: }
//     CHECK-NOT: vector.transfer_write
// Barrier after the loop
//         CHECK: gpu.barrier

//     CHECK-NOT: vector.transfer_read
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}vector.broadcast{{.*}}{{[[:space:]].*}}vector.bitcast{{.*}}{{[[:space:]].*}}arith.addi{{.*}}vector<4xi8>
//         CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<32xvector<4xi8>, #gpu.address_space<workgroup>>
//         CHECK: scf.if
//         CHECK:   memref.store %{{.*}}, %[[ALLOC]][%{{.*}}] : memref<32xvector<4xi8>, #gpu.address_space<workgroup>>
//         CHECK: }
// CHECK-COUNT-5: gpu.shuffle  xor{{.*}}{{[[:space:]].*}}vector.broadcast{{.*}}{{[[:space:]].*}}vector.bitcast{{.*}}{{[[:space:]].*}}arith.addi{{.*}}vector<4xi8>

//         CHECK: %[[RES:.*]] = vector.reduction <add>, %59 : vector<4xi8> into i8
//         CHECK: %[[RES_VEC:.*]] = vector.broadcast %[[RES]] : i8 to vector<1xi8>
//         CHECK: %[[CONDXIS0:.*]] = arith.cmpi eq, %[[TIDX]], %[[C0]] : index
//         CHECK: scf.if %[[CONDXIS0]]
//         CHECK:   vector.transfer_write %[[RES_VEC]], %[[ALLOC0]][%[[C0]]] {in_bounds = [true]} : vector<1xi8>, memref<1xi8, #gpu.address_space<workgroup>>

//         CHECK:   gpu.barrier
//         CHECK:   arith.divui {{.*}} vector<8xi8>
//         CHECK:   arith.divui {{.*}} i8
//         CHECK:   gpu.barrier
