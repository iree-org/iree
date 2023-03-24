// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target, fold-memref-alias-ops, canonicalize, cse)))" %s | FileCheck %s

#device_target_cuda = #hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>]}>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>
module attributes {hal.device.targets = [#device_target_cuda]} {
  hal.executable @transpose_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {
      hal.executable.export public @transpose_dispatch_0_generic_4096x4096 ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @transpose_dispatch_0_generic_4096x4096() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4096x4096xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4096x4096xf32>> -> tensor<4096x4096xf32>
          %3 = tensor.empty() : tensor<4096x4096xf32>
          %4 = linalg.generic {indexing_maps = [ affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<4096x4096xf32>) outs(%3 : tensor<4096x4096xf32>) {
          ^bb0(%arg0: f32, %arg1: f32):
            linalg.yield %arg0 : f32
          } -> tensor<4096x4096xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : tensor<4096x4096xf32> -> !flow.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
          return
        }
      }
    }
  }
}

// CHECK-LABEL:  hal.executable public @transpose_dispatch_0
//   CHECK-DAG:  %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:  %[[D0:.*]] = gpu.thread_id  x
//   CHECK-DAG:  %[[D1:.*]] = gpu.thread_id  y
//   CHECK-DAG:  %[[D2:.*]] = gpu.thread_id  z
//   CHECK-DAG:  %[[D3:.*]] = memref.alloc() : memref<32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:  %[[D4:.*]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[C0]]) : memref<4096x4096xf32>
//       CHECK:  memref.assume_alignment %[[D4]], 64 : memref<4096x4096xf32>
//       CHECK:  %[[D5:.*]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%[[C0]]) : memref<4096x4096xf32>
//       CHECK:  memref.assume_alignment %[[D5]], 64 : memref<4096x4096xf32>
//       CHECK:  gpu.barrier
//       CHECK:  %[[D6:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]], %[[D1]], %[[D2]], %{{.*}}]
//       CHECK:  %[[D7:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]], %{{.*}}]
//       CHECK:  %[[D8:.*]] = vector.transfer_read %[[D4]]{{\[}}%[[D6]], %[[D7]]], %[[CST]] {in_bounds = [true, true]} : memref<4096x4096xf32>, vector<1x4xf32>
//       CHECK:  %[[D9:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]], %[[D1]], %[[D2]]]
//       CHECK:  %[[D10:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]]]
//       CHECK:  vector.transfer_write %[[D8]], %[[D3]]{{\[}}%[[D9]], %[[D10]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:  gpu.barrier
//       CHECK:  %[[D11:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]]]
//       CHECK:  %[[D12:.*]] = vector.transfer_read %[[D3]]{{\[}}%[[D11]], %[[D1]]], %[[CST]] {in_bounds = [true, true]} : memref<32x33xf32, #gpu.address_space<workgroup>>, vector<4x1xf32>
//       CHECK:  %[[D13:.*]] = vector.shape_cast %[[D12]] : vector<4x1xf32> to vector<1x4xf32>
//       CHECK:  %[[D14:.*]] = vector.extract %[[D13]][0] : vector<1x4xf32>
//       CHECK:  %[[D15:.*]] = affine.apply #{{.*}}(){{\[}}%{{.*}}, %[[D1]]]
//       CHECK:  %[[D16:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]], %{{.*}}]
//       CHECK:  vector.transfer_write %[[D14]], %[[D5]]{{\[}}%[[D15]], %[[D16]]] {in_bounds = [true]} : vector<4xf32>, memref<4096x4096xf32>

// -----


#device_target_cuda = #hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>]}>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>
module attributes {hal.device.targets = [#device_target_cuda]} {
  hal.executable @transpose_single_operand_dispatch_0_generic_768x2048 {
    hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {
      hal.executable.export public @transpose_single_operand_dispatch_0_generic_768x2048 ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @transpose_single_operand_dispatch_0_generic_768x2048() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2048x768xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<768x2048xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<768x2048xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x768xf32>> -> tensor<2048x768xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [768, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<768x2048xf32>> -> tensor<768x2048xf32>
          %5 = tensor.empty() : tensor<768x2048xf32>
          %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3, %4 : tensor<2048x768xf32>, tensor<768x2048xf32>) outs(%5 : tensor<768x2048xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
            %7 = arith.addf %arg0, %arg1 : f32
            linalg.yield %7 : f32
          } -> tensor<768x2048xf32>
          flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [768, 2048], strides = [1, 1] : tensor<768x2048xf32> -> !flow.dispatch.tensor<writeonly:tensor<768x2048xf32>>
          return
        }
      }
    }
  }
}

// CHECK-LABEL:  hal.executable public @transpose_single_operand_dispatch_0_generic_768x2048
//       CHECK:  %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK:  %[[C0:.*]] = arith.constant 0 : index
//       CHECK:  %[[D0:.*]] = gpu.thread_id  x
//       CHECK:  %[[D1:.*]] = gpu.thread_id  y
//       CHECK:  %[[D2:.*]] = gpu.thread_id  z
//       CHECK:  %[[D3:.*]] = memref.alloc() : memref<32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:  %[[D4:.*]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[C0]]) : memref<2048x768xf32>
//       CHECK:  memref.assume_alignment %[[D4]], 64 : memref<2048x768xf32>
//       CHECK:  %[[D5:.*]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%[[C0]]) : memref<768x2048xf32>
//       CHECK:  memref.assume_alignment %[[D5]], 64 : memref<768x2048xf32>
//       CHECK:  %[[D6:.*]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%[[C0]]) : memref<768x2048xf32>
//       CHECK:  memref.assume_alignment %[[D6]], 64 : memref<768x2048xf32>
//       CHECK:  gpu.barrier
//       CHECK:  %[[D7:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]], %[[D1]], %[[D2]], %{{.*}}]
//       CHECK:  %[[D8:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]], %{{.*}}]
//       CHECK:  %[[D9:.*]] = vector.transfer_read %[[D4]]{{\[}}%[[D7]], %[[D8]]], %[[CST]] {in_bounds = [true, true]} : memref<2048x768xf32>, vector<1x4xf32>
//       CHECK:  %[[D10:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]], %[[D1]], %[[D2]]]
//       CHECK:  %[[D11:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]]]
//       CHECK:  vector.transfer_write %[[D9]], %[[D3]]{{\[}}%[[D10]], %[[D11]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:  gpu.barrier
//       CHECK:  %[[D12:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]]]
//       CHECK:  %[[D13:.*]] = vector.transfer_read %[[D3]]{{\[}}%[[D12]], %[[D1]]], %[[CST]] {in_bounds = [true, true]} : memref<32x33xf32, #gpu.address_space<workgroup>>, vector<4x1xf32>
//       CHECK:  %[[D14:.*]] = vector.shape_cast %[[D13]] : vector<4x1xf32> to vector<1x4xf32>
//       CHECK:  %[[D15:.*]] = affine.apply #{{.*}}(){{\[}}%{{.*}}, %[[D1]]]
//       CHECK:  %[[D16:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]], %{{.*}}]
//       CHECK:  %[[D17:.*]] = vector.transfer_read %[[D5]]{{\[}}%[[D15]], %[[D16]]], %[[CST]] {in_bounds = [true]} : memref<768x2048xf32>, vector<4xf32>
//       CHECK:  %[[D18:.*]] = vector.extract %[[D14]][0] : vector<1x4xf32>
//       CHECK:  %[[D19:.*]] = arith.addf %[[D18]], %[[D17]] : vector<4xf32>
//       CHECK:  vector.transfer_write %[[D19]], %[[D6]]{{\[}}%[[D15]], %[[D16]]] {in_bounds = [true]} : vector<4xf32>, memref<768x2048xf32>

// -----

#device_target_cuda = #hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>]}>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>
module attributes {hal.device.targets = [#device_target_cuda]} {
  hal.executable @transpose_3d_no_dispatch_0_generic_768x2048x1024 {
    hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {
      hal.executable.export public @transpose_3d_no_dispatch_0_generic_768x2048x1024 ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @transpose_3d_no_dispatch_0_generic_768x2048x1024() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2048x768x1024xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<768x2048x1024xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<768x2048x1024xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2048, 768, 1024], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x768x1024xf32>> -> tensor<2048x768x1024xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [768, 2048, 1024], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<768x2048x1024xf32>> -> tensor<768x2048x1024xf32>
          %5 = tensor.empty() : tensor<768x2048x1024xf32>
          %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<2048x768x1024xf32>, tensor<768x2048x1024xf32>) outs(%5 : tensor<768x2048x1024xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
            %7 = arith.addf %arg0, %arg1 : f32
            linalg.yield %7 : f32
          } -> tensor<768x2048x1024xf32>
          flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [768, 2048, 1024], strides = [1, 1, 1] : tensor<768x2048x1024xf32> -> !flow.dispatch.tensor<writeonly:tensor<768x2048x1024xf32>>
          return
        }
      }
    }
  }
}

// CHECK-LABEL:   hal.executable public @transpose_3d_no_dispatch_0_generic_768x2048x1024 {
//   CHECK-NOT:   gpu.barrier
//   CHECK-NOT:   memref.alloc
//       CHECK:   return

// -----

#device_target_cuda = #hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>]}>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>
module attributes {hal.device.targets = [#device_target_cuda]} {
  hal.executable @transpose_3d_yes_dispatch_0_generic_10x768x2048 {
    hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {
      hal.executable.export public @transpose_3d_yes_dispatch_0_generic_10x768x2048 ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @transpose_3d_yes_dispatch_0_generic_10x768x2048() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10x2048x768xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10x768x2048xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<10x768x2048xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [10, 2048, 768], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<10x2048x768xf32>> -> tensor<10x2048x768xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [10, 768, 2048], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<10x768x2048xf32>> -> tensor<10x768x2048xf32>
          %5 = tensor.empty() : tensor<10x768x2048xf32>
          %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<10x2048x768xf32>, tensor<10x768x2048xf32>) outs(%5 : tensor<10x768x2048xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
            %7 = arith.addf %arg0, %arg1 : f32
            linalg.yield %7 : f32
          } -> tensor<10x768x2048xf32>
          flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [10, 768, 2048], strides = [1, 1, 1] : tensor<10x768x2048xf32> -> !flow.dispatch.tensor<writeonly:tensor<10x768x2048xf32>>
          return
        }
      }
    }
  }
}

// CHECK-LABEL:   hal.executable public @transpose_3d_yes_dispatch_0_generic_10x768x2048 {
//   CHECK-DAG:   %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[D0:.*]] = gpu.thread_id  x
//       CHECK:   %[[D1:.*]] = gpu.thread_id  y
//       CHECK:   %[[D2:.*]] = gpu.thread_id  z
//       CHECK:   %[[D3:.*]] = memref.alloc() : memref<1x32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[D4:.*]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[C0]]) : memref<10x2048x768xf32>
//       CHECK:   memref.assume_alignment %[[D4]], 64 : memref<10x2048x768xf32>
//       CHECK:   %[[D5:.*]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%[[C0]]) : memref<10x768x2048xf32>
//       CHECK:   memref.assume_alignment %[[D5]], 64 : memref<10x768x2048xf32>
//       CHECK:   %[[D6:.*]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%[[C0]]) : memref<10x768x2048xf32>
//       CHECK:   memref.assume_alignment %[[D6]], 64 : memref<10x768x2048xf32>
//       CHECK:   gpu.barrier
//       CHECK:   %[[D7:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]], %[[D1]], %[[D2]], %{{.*}}]
//       CHECK:   %[[D8:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]], %{{.*}}]
//       CHECK:   %[[D9:.*]] = vector.transfer_read %[[D4]]{{\[}}%{{.*}}, %[[D7]], %[[D8]]], %[[CST]] {in_bounds = [true, true, true]} : memref<10x2048x768xf32>, vector<1x1x4xf32>
//       CHECK:   %[[D10:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]], %[[D1]], %[[D2]]]
//       CHECK:   %[[D11:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]]]
//       CHECK:   vector.transfer_write %[[D9]], %[[D3]]{{\[}}%[[C0]], %[[D10]], %[[D11]]] {in_bounds = [true, true, true]} : vector<1x1x4xf32>, memref<1x32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:   gpu.barrier
//       CHECK:   %[[D12:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]]]
//       CHECK:   %[[D13:.*]] = vector.transfer_read %[[D3]]{{\[}}%[[C0]], %[[D12]], %[[D1]]], %[[CST]] {in_bounds = [true, true]} : memref<1x32x33xf32, #gpu.address_space<workgroup>>, vector<4x1xf32>
//       CHECK:   %[[D14:.*]] = vector.broadcast %[[D13]] : vector<4x1xf32> to vector<1x4x1xf32>
//       CHECK:   %[[D15:.*]] = vector.shape_cast %[[D14]] : vector<1x4x1xf32> to vector<1x1x4xf32>
//       CHECK:   %[[D16:.*]] = affine.apply #{{.*}}(){{\[}}%{{.*}}, %[[D1]]]
//       CHECK:   %[[D17:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]], %{{.*}}]
//       CHECK:   %[[D18:.*]] = vector.transfer_read %[[D5]]{{\[}}%{{.*}}, %[[D16]], %[[D17]]], %[[CST]] {in_bounds = [true]} : memref<10x768x2048xf32>, vector<4xf32>
//       CHECK:   %[[D19:.*]] = vector.extract %[[D15]][0, 0] : vector<1x1x4xf32>
//       CHECK:   %[[D20:.*]] = arith.addf %[[D19]], %[[D18]] : vector<4xf32>
//       CHECK:   vector.transfer_write %[[D20]], %[[D6]]{{\[}}%{{.*}}, %[[D16]], %[[D17]]] {in_bounds = [true]} : vector<4xf32>, memref<10x768x2048xf32>

// -----

#device_target_cuda = #hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>]}>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>
module attributes {hal.device.targets = [#device_target_cuda]} {
  hal.executable @transpose_3d_trans_out_dispatch_0_generic_10x2048x768 {
    hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {
      hal.executable.export public @transpose_3d_trans_out_dispatch_0_generic_10x2048x768 ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @transpose_3d_trans_out_dispatch_0_generic_10x2048x768() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10x768x2048xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10x768x2048xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<10x2048x768xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [10, 768, 2048], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<10x768x2048xf32>> -> tensor<10x768x2048xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [10, 768, 2048], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<10x768x2048xf32>> -> tensor<10x768x2048xf32>
          %5 = tensor.empty() : tensor<10x2048x768xf32>
          %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<10x768x2048xf32>, tensor<10x768x2048xf32>) outs(%5 : tensor<10x2048x768xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
            %7 = arith.addf %arg0, %arg1 : f32
            linalg.yield %7 : f32
          } -> tensor<10x2048x768xf32>
          flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [10, 2048, 768], strides = [1, 1, 1] : tensor<10x2048x768xf32> -> !flow.dispatch.tensor<writeonly:tensor<10x2048x768xf32>>
          return
        }
      }
    }
  }
}

// CHECK-LABEL:   hal.executable public @transpose_3d_trans_out_dispatch_0_generic_10x2048x768 {
//   CHECK-DAG:   %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[D0:.*]] = gpu.thread_id  x
//       CHECK:   %[[D1:.*]] = gpu.thread_id  y
//       CHECK:   %[[D2:.*]] = gpu.thread_id  z
//       CHECK:   %[[D3:.*]] = memref.alloc() : memref<1x32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[D4:.*]] = memref.alloc() : memref<1x32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[D5:.*]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[C0]]) : memref<10x768x2048xf32>
//       CHECK:   memref.assume_alignment %[[D5]], 64 : memref<10x768x2048xf32>
//       CHECK:   %[[D6:.*]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%[[C0]]) : memref<10x768x2048xf32>
//       CHECK:   memref.assume_alignment %[[D6]], 64 : memref<10x768x2048xf32>
//       CHECK:   %[[D7:.*]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%[[C0]]) : memref<10x2048x768xf32>
//       CHECK:   memref.assume_alignment %[[D7]], 64 : memref<10x2048x768xf32>
//       CHECK:   gpu.barrier
//       CHECK:   %[[D8:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]], %[[D1]], %[[D2]], %{{.*}}]
//       CHECK:   %[[D9:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]], %{{.*}}]
//       CHECK:   %[[D10:.*]] = vector.transfer_read %[[D5]]{{\[}}%{{.*}}, %[[D8]], %[[D9]]], %[[CST]] {in_bounds = [true, true, true]} : memref<10x768x2048xf32>, vector<1x1x4xf32>
//       CHECK:   %[[D11:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]], %[[D1]], %[[D2]]]
//       CHECK:   %[[D12:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]]]
//       CHECK:   vector.transfer_write %[[D10]], %[[D4]]{{\[}}%[[C0]], %[[D11]], %[[D12]]] {in_bounds = [true, true, true]} : vector<1x1x4xf32>, memref<1x32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:   %[[D13:.*]] = vector.transfer_read %[[D6]]{{\[}}%{{.*}}, %[[D8]], %[[D9]]], %[[CST]] {in_bounds = [true, true, true]} : memref<10x768x2048xf32>, vector<1x1x4xf32>
//       CHECK:   vector.transfer_write %[[D13]], %[[D3]]{{\[}}%[[C0]], %[[D11]], %[[D12]]] {in_bounds = [true, true, true]} : vector<1x1x4xf32>, memref<1x32x33xf32, #gpu.address_space<workgroup>>
//       CHECK:   gpu.barrier
//       CHECK:   %[[D14:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]]]
//       CHECK:   %[[D15:.*]] = vector.transfer_read %[[D4]]{{\[}}%[[C0]], %[[D14]], %[[D1]]], %[[CST]] {in_bounds = [true, true]} : memref<1x32x33xf32, #gpu.address_space<workgroup>>, vector<4x1xf32>
//       CHECK:   %[[D16:.*]] = vector.transfer_read %[[D3]]{{\[}}%[[C0]], %[[D14]], %[[D1]]], %[[CST]] {in_bounds = [true, true]} : memref<1x32x33xf32, #gpu.address_space<workgroup>>, vector<4x1xf32>
//       CHECK:   %[[D17:.*]] = arith.addf %[[D15]], %[[D16]] : vector<4x1xf32>
//       CHECK:   %[[D18:.*]] = vector.broadcast %[[D17]] : vector<4x1xf32> to vector<1x4x1xf32>
//       CHECK:   %[[D19:.*]] = vector.shape_cast %[[D18]] : vector<1x4x1xf32> to vector<1x1x4xf32>
//       CHECK:   %[[D20:.*]] = vector.extract %[[D19]][0, 0] : vector<1x1x4xf32>
//       CHECK:   %[[D21:.*]] = affine.apply #{{.*}}(){{\[}}%{{.*}}, %[[D1]]]
//       CHECK:   %[[D22:.*]] = affine.apply #{{.*}}(){{\[}}%[[D0]], %{{.*}}]
//       CHECK:   vector.transfer_write %[[D20]], %[[D7]]{{\[}}%{{.*}}, %[[D21]], %[[D22]]] {in_bounds = [true]} : vector<4xf32>, memref<10x2048x768xf32>

// -----

#device_target_cuda = #hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>]}>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>
module attributes {hal.device.targets = [#device_target_cuda]} {
  hal.executable @transpose_3d_diff_dispatch_0_generic_10x768x2048 {
    hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {
      hal.executable.export public @transpose_3d_diff_dispatch_0_generic_10x768x2048 ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @transpose_3d_diff_dispatch_0_generic_10x768x2048() {
        %c256 = arith.constant 256 : index
        %c10 = arith.constant 10 : index
        %c768 = arith.constant 768 : index
        %c2048 = arith.constant 2048 : index
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<10x2048x768xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2048x768x10xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<10x768x2048xf32>>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %workgroup_count_x = hal.interface.workgroup.count[0] : index
        %workgroup_id_y = hal.interface.workgroup.id[1] : index
        %workgroup_count_y = hal.interface.workgroup.count[1] : index
        %workgroup_id_z = hal.interface.workgroup.id[2] : index
        %workgroup_count_z = hal.interface.workgroup.count[2] : index
        scf.for %arg0 = %workgroup_id_z to %c10 step %workgroup_count_z {
            scf.for %arg1 = %workgroup_id_y to %c768 step %workgroup_count_y {
            %3 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_id_x]
            %4 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%workgroup_count_x]
            scf.for %arg2 = %3 to %c2048 step %4 {
                %5 = flow.dispatch.tensor.load %0, offsets = [%arg0, %arg2, %arg1], sizes = [1, %c256, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<10x2048x768xf32>> -> tensor<1x?x1xf32>
                %6 = flow.dispatch.tensor.load %1, offsets = [%arg2, %arg1, %arg0], sizes = [%c256, 1, 1], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x768x10xf32>> -> tensor<?x1x1xf32>
                %7 = tensor.empty() : tensor<1x1x256xf32>
                %8 = tensor.cast %5 : tensor<1x?x1xf32> to tensor<1x256x1xf32>
                %9 = tensor.cast %6 : tensor<?x1x1xf32> to tensor<256x1x1xf32>
                %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d2, d1, d0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %9 : tensor<1x256x1xf32>, tensor<256x1x1xf32>) outs(%7 : tensor<1x1x256xf32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 256]]>} {
                ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
                %12 = arith.addf %arg3, %arg4 : f32
                linalg.yield %12 : f32
                } -> tensor<1x1x256xf32>
                %11 = tensor.cast %10 : tensor<1x1x256xf32> to tensor<1x1x?xf32>
                flow.dispatch.tensor.store %11, %2, offsets = [%arg0, %arg1, %arg2], sizes = [1, 1, %c256], strides = [1, 1, 1] : tensor<1x1x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<10x768x2048xf32>>
            }
            }
        }
        return
        }
      }
    }
  }
}

// CHECK-LABEL:   hal.executable public @transpose_3d_diff_dispatch_0_generic_10x768x2048 {
//   CHECK-NOT:   gpu.barrier
//   CHECK-NOT:   memref.alloc
//       CHECK:   return
