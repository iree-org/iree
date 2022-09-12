// RUN: iree-opt --split-input-file --pass-pipeline='hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target-pass))' %s --fold-memref-alias-ops -canonicalize -cse | FileCheck %s

#device_target_cuda = #hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>], legacy_sync}>
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
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:4096x4096xf32>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:4096x4096xf32>
          %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !flow.dispatch.tensor<readonly:4096x4096xf32> -> tensor<4096x4096xf32>
          %3 = linalg.init_tensor [4096, 4096] : tensor<4096x4096xf32>
          %4 = linalg.generic {indexing_maps = [ affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2 : tensor<4096x4096xf32>) outs(%3 : tensor<4096x4096xf32>) {
          ^bb0(%arg0: f32, %arg1: f32):
            linalg.yield %arg0 : f32
          } -> tensor<4096x4096xf32>
          flow.dispatch.tensor.store %4, %1, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : tensor<4096x4096xf32> -> !flow.dispatch.tensor<writeonly:4096x4096xf32>
          return
        }
      }
    }
  }
}

// CHECK-LABEL:  hal.executable public @transpose_dispatch_0
//       CHECK:  hal.executable.variant public @cuda
//   CHECK-DAG:  %[[C0:.+]] = arith.constant 0
//   CHECK-DAG:  %[[CST:.+]] = arith.constant 0
//   CHECK-DAG:  %[[IN:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:  %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:  %[[ALLOC:.+]] = memref.alloc() : memref<32x33xf32, 3>
//       CHECK:  gpu.barrier
//       CHECK:  %[[R0:.+]] = vector.transfer_read %[[IN]][%{{.*}}, %{{.*}}], %[[CST]] {in_bounds = [true]} : memref<4096x4096xf32>, vector<4xf32>
//       CHECK:  vector.transfer_write %[[R0]], %[[ALLOC]][%{{.*}}, %{{.*}}] {in_bounds = [true]} : vector<4xf32>, memref<32x33xf32, 3>
//       CHECK:  gpu.barrier
//       CHECK:  %[[R1:.+]] = vector.transfer_read %[[ALLOC]][%{{.*}}, %{{.*}}], %[[CST]] {in_bounds = [true, true]} : memref<32x33xf32, 3>, vector<4x1xf32>
//       CHECK:  %[[R2:.+]] = vector.shape_cast %[[R1]] : vector<4x1xf32> to vector<1x4xf32>
//       CHECK:  %[[R3:.+]] = vector.extract %[[R2]][0] : vector<1x4xf32>
//       CHECK:  vector.transfer_write %[[R3]], %[[OUT]][%{{.*}}, %{{.*}}] {in_bounds = [true]} : vector<4xf32>, memref<4096x4096xf32>

// -----

#device_target_cuda = #hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>], legacy_sync}>
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
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:2048x768xf32>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:768x2048xf32>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:768x2048xf32>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 768], strides = [1, 1] : !flow.dispatch.tensor<readonly:2048x768xf32> -> tensor<2048x768xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [768, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:768x2048xf32> -> tensor<768x2048xf32>
          %5 = linalg.init_tensor [768, 2048] : tensor<768x2048xf32>
          %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3, %4 : tensor<2048x768xf32>, tensor<768x2048xf32>) outs(%5 : tensor<768x2048xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
            %7 = arith.addf %arg0, %arg1 : f32
            linalg.yield %7 : f32
          } -> tensor<768x2048xf32>
          flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [768, 2048], strides = [1, 1] : tensor<768x2048xf32> -> !flow.dispatch.tensor<writeonly:768x2048xf32>
          return
        }
      }
    }
  }
}

// CHECK-LABEL:  hal.executable public @transpose_single_operand_dispatch_0_generic_768x2048
//       CHECK:  hal.executable.variant public @cuda
//   CHECK-DAG:  %[[C0:.+]] = arith.constant 0
//   CHECK-DAG:  %[[CST:.+]] = arith.constant 0
//   CHECK-DAG:  %[[ALLOC:.+]] = memref.alloc() : memref<32x33xf32, 3>
//   CHECK-DAG:  %[[IN0:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:  %[[IN1:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:  %[[OUT:.+]] = hal.interface.binding.subspan set(0) binding(2)
//       CHECK:  gpu.barrier
//       CHECK:  %[[R0:.+]] = vector.transfer_read %[[IN0]][%{{.*}}, %{{.*}}], %[[CST]] {in_bounds = [true]} : memref<2048x768xf32>, vector<4xf32>
//       CHECK:  vector.transfer_write %[[R0]], %[[ALLOC]][%{{.*}}, %{{.*}}] {in_bounds = [true]} : vector<4xf32>, memref<32x33xf32, 3>
//       CHECK:  gpu.barrier
//       CHECK:  %[[D17:.+]] = memref.subview %[[D4]][%{{.*}}, %{{.*}}] [4, 1] [1, 1] : memref<32x32xf32, #{{.*}}, 3> to memref<4x1xf32, #{{.*}}, 3>
//       CHECK:  %[[D18:.+]] = memref.subview %{{.*}}[%{{.*}}, %{{.*}}] [1, 4] [1, 1] : memref<32x32xf32, #{{.*}}> to memref<1x4xf32, #{{.*}}>
//       CHECK:  %[[D19:.+]] = memref.subview %{{.*}}[%{{.*}}, %{{.*}}] [1, 4] [1, 1] : memref<32x32xf32, #{{.*}}> to memref<1x4xf32, #{{.*}}>
//       CHECK:  %[[D20:.+]] = vector.transfer_read %[[D17]][%{{.*}}, %{{.*}}], %[[CST]] {in_bounds = [true, true]} : memref<4x1xf32, #{{.*}}, 3>, vector<4x1xf32>
//       CHECK:  %[[D21:.+]] = vector.shape_cast %[[D20]] : vector<4x1xf32> to vector<1x4xf32>
//       CHECK:  %[[D22:.+]] = vector.transfer_read %[[D18]][%{{.*}}, %{{.*}}], %[[CST]] {in_bounds = [true]} : memref<1x4xf32, #{{.*}}>, vector<4xf32>
//       CHECK:  %[[D23:.+]] = vector.extract %[[D21]][0] : vector<1x4xf32>
//       CHECK:  %[[D24:.+]] = arith.addf %[[D23]], %[[D22]] : vector<4xf32>
//       CHECK:  vector.transfer_write %[[D24]], %[[D19]][%{{.*}}, %{{.*}}] {in_bounds = [true]} : vector<4xf32>, memref<1x4xf32, #{{.*}}>

// -----

#device_target_cuda = #hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>], legacy_sync}>
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
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:2048x768x1024xf32>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:768x2048x1024xf32>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:768x2048x1024xf32>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2048, 768, 1024], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:2048x768x1024xf32> -> tensor<2048x768x1024xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [768, 2048, 1024], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:768x2048x1024xf32> -> tensor<768x2048x1024xf32>
          %5 = linalg.init_tensor [768, 2048, 1024] : tensor<768x2048x1024xf32>
          %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<2048x768x1024xf32>, tensor<768x2048x1024xf32>) outs(%5 : tensor<768x2048x1024xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
            %7 = arith.addf %arg0, %arg1 : f32
            linalg.yield %7 : f32
          } -> tensor<768x2048x1024xf32>
          flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [768, 2048, 1024], strides = [1, 1, 1] : tensor<768x2048x1024xf32> -> !flow.dispatch.tensor<writeonly:768x2048x1024xf32>
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

#device_target_cuda = #hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>], legacy_sync}>
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
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:10x2048x768xf32>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:10x768x2048xf32>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:10x768x2048xf32>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [10, 2048, 768], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:10x2048x768xf32> -> tensor<10x2048x768xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [10, 768, 2048], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:10x768x2048xf32> -> tensor<10x768x2048xf32>
          %5 = linalg.init_tensor [10, 768, 2048] : tensor<10x768x2048xf32>
          %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<10x2048x768xf32>, tensor<10x768x2048xf32>) outs(%5 : tensor<10x768x2048xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
            %7 = arith.addf %arg0, %arg1 : f32
            linalg.yield %7 : f32
          } -> tensor<10x768x2048xf32>
          flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [10, 768, 2048], strides = [1, 1, 1] : tensor<10x768x2048xf32> -> !flow.dispatch.tensor<writeonly:10x768x2048xf32>
          return
        }
      }
    }
  }
}

// CHECK-LABEL:   hal.executable public @transpose_3d_yes_dispatch_0_generic_10x768x2048 {
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[D0:.*]] = gpu.thread_id  x
//       CHECK:   %[[D1:.*]] = gpu.thread_id  y
//       CHECK:   %[[D2:.*]] = gpu.thread_id  z
//       CHECK:   %[[D3:.*]] = memref.alloc() : memref<1x32x33xf32, 3>
//       CHECK:   %[[D4:.*]] = memref.subview %[[D3]][0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<1x32x33xf32, 3> to memref<1x32x32xf32, #{{.*}}1, 3>
//       CHECK:   %[[D5:.*]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) alignment(64) : memref<10x2048x768xf32>
//       CHECK:   memref.assume_alignment %[[D5]], 64 : memref<10x2048x768xf32>
//       CHECK:   %[[D6:.*]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%[[C0]]) alignment(64) : memref<10x768x2048xf32>
//       CHECK:   memref.assume_alignment %[[D6]], 64 : memref<10x768x2048xf32>
//       CHECK:   %[[D7:.*]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%[[C0]]) alignment(64) : memref<10x768x2048xf32>
//       CHECK:   memref.assume_alignment %[[D7]], 64 : memref<10x768x2048xf32>
//       CHECK:   %[[D10:.*]] = memref.subview %[[D7]]{{\[}}%{{.*}}, %{{.*}}, %{{.*}}] [1, 32, 32] [1, 1, 1] : memref<10x768x2048xf32> to memref<1x32x32xf32, #{{.*}}>
//       CHECK:   %[[D11:.*]] = memref.subview %[[D5]]{{\[}}%{{.*}}, %{{.*}}, %{{.*}}] [1, 32, 32] [1, 1, 1] : memref<10x2048x768xf32> to memref<1x32x32xf32, #{{.*}}>
//       CHECK:   %[[D12:.*]] = memref.subview %[[D6]]{{\[}}%{{.*}}, %{{.*}}, %{{.*}}] [1, 32, 32] [1, 1, 1] : memref<10x768x2048xf32> to memref<1x32x32xf32, #{{.*}}>
//       CHECK:   gpu.barrier
//       CHECK:   %[[D13:.*]] = affine.apply #{{.*}}5(){{\[}}%[[D0]], %[[D1]], %[[D2]]]
//       CHECK:   %[[D14:.*]] = affine.apply #{{.*}}6(){{\[}}%[[D0]]]
//       CHECK:   %[[D15:.*]] = vector.transfer_read %[[D11]]{{\[}}%[[C0]], %[[D13]], %[[D14]]], %[[CST]] {in_bounds = [true]} : memref<1x32x32xf32, #{{.*}}>, vector<4xf32>
//       CHECK:   vector.transfer_write %[[D15]], %[[D4]]{{\[}}%[[C0]], %[[D13]], %[[D14]]] {in_bounds = [true]} : vector<4xf32>, memref<1x32x32xf32, #{{.*}}, 3>
//       CHECK:   gpu.barrier
//       CHECK:   %[[D16:.*]] = affine.apply #{{.*}}7(){{\[}}%[[D0]]]
//       CHECK:   %[[D17:.*]] = memref.subview %[[D4]][0, %[[D16]], %[[D1]]] [1, 4, 1] [1, 1, 1] : memref<1x32x32xf32, #{{.*}}1, 3> to memref<1x4x1xf32, #{{.*}}, 3>
//       CHECK:   %[[D18:.*]] = memref.subview %[[D12]][0, %[[D1]], %[[D16]]] [1, 1, 4] [1, 1, 1] : memref<1x32x32xf32, #{{.*}}3> to memref<1x1x4xf32, #{{.*}}>
//       CHECK:   %[[D19:.*]] = memref.subview %[[D10]][0, %[[D1]], %[[D16]]] [1, 1, 4] [1, 1, 1] : memref<1x32x32xf32, #{{.*}}3> to memref<1x1x4xf32, #{{.*}}>
//       CHECK:   %[[D20:.*]] = vector.transfer_read %[[D17]]{{\[}}%[[C0]], %[[C0]], %[[C0]]], %[[CST]] {in_bounds = [true, true]} : memref<1x4x1xf32, #{{.*}}, 3>, vector<4x1xf32>
//       CHECK:   %[[D21:.*]] = vector.broadcast %[[D20]] : vector<4x1xf32> to vector<1x4x1xf32>
//       CHECK:   %[[D22:.*]] = vector.shape_cast %[[D21]] : vector<1x4x1xf32> to vector<1x1x4xf32>
//       CHECK:   %[[D23:.*]] = vector.transfer_read %[[D18]]{{\[}}%[[C0]], %[[C0]], %[[C0]]], %[[CST]] {in_bounds = [true]} : memref<1x1x4xf32, #{{.*}}3>, vector<4xf32>
//       CHECK:   %[[D24:.*]] = vector.extract %[[D22]][0, 0] : vector<1x1x4xf32>
//       CHECK:   %[[D25:.*]] = arith.addf %[[D24]], %[[D23]] : vector<4xf32>
//       CHECK:   vector.transfer_write %[[D25]], %[[D19]]{{\[}}%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true]} : vector<4xf32>, memref<1x1x4xf32, #{{.*}}3>

// -----

#device_target_cuda = #hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_80"}>], legacy_sync}>
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
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:10x768x2048xf32>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:10x768x2048xf32>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<writeonly:10x2048x768xf32>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [10, 768, 2048], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:10x768x2048xf32> -> tensor<10x768x2048xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [10, 768, 2048], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:10x768x2048xf32> -> tensor<10x768x2048xf32>
          %5 = linalg.init_tensor [10, 2048, 768] : tensor<10x2048x768xf32>
          %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%3, %4 : tensor<10x768x2048xf32>, tensor<10x768x2048xf32>) outs(%5 : tensor<10x2048x768xf32>) {
          ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
            %7 = arith.addf %arg0, %arg1 : f32
            linalg.yield %7 : f32
          } -> tensor<10x2048x768xf32>
          flow.dispatch.tensor.store %6, %2, offsets = [0, 0, 0], sizes = [10, 2048, 768], strides = [1, 1, 1] : tensor<10x2048x768xf32> -> !flow.dispatch.tensor<writeonly:10x2048x768xf32>
          return
        }
      }
    }
  }
}

// CHECK-LABEL:   hal.executable public @transpose_3d_trans_out_dispatch_0_generic_10x2048x768 {
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[CST:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK:   %[[D0:.*]] = gpu.thread_id  x
//       CHECK:   %[[D1:.*]] = gpu.thread_id  y
//       CHECK:   %[[D2:.*]] = gpu.thread_id  z
//       CHECK:   %[[D3:.*]] = memref.alloc() : memref<1x32x33xf32, 3>
//       CHECK:   %[[D4:.*]] = memref.subview %[[D3]][0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<1x32x33xf32, 3> to memref<1x32x32xf32, #{{.*}}, 3>
//       CHECK:   %[[D5:.*]] = memref.alloc() : memref<1x32x33xf32, 3>
//       CHECK:   %[[D6:.*]] = memref.subview %[[D5]][0, 0, 0] [1, 32, 32] [1, 1, 1] : memref<1x32x33xf32, 3> to memref<1x32x32xf32, #{{.*}}, 3>
//       CHECK:   %[[D7:.*]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%[[C0]]) alignment(64) : memref<10x768x2048xf32>
//       CHECK:   memref.assume_alignment %[[D7]], 64 : memref<10x768x2048xf32>
//       CHECK:   %[[D8:.*]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%[[C0]]) alignment(64) : memref<10x768x2048xf32>
//       CHECK:   memref.assume_alignment %[[D8]], 64 : memref<10x768x2048xf32>
//       CHECK:   %[[D9:.*]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset(%[[C0]]) alignment(64) : memref<10x2048x768xf32>
//       CHECK:   memref.assume_alignment %[[D9]], 64 : memref<10x2048x768xf32>
//       CHECK:   %[[D12:.*]] = memref.subview %[[D9]]{{\[}}%{{.*}}, %{{.*}}, %{{.*}}] [1, 32, 32] [1, 1, 1] : memref<10x2048x768xf32> to memref<1x32x32xf32, #{{.*}}>
//       CHECK:   %[[D13:.*]] = memref.subview %[[D7]]{{\[}}%{{.*}}, %{{.*}}, %{{.*}}] [1, 32, 32] [1, 1, 1] : memref<10x768x2048xf32> to memref<1x32x32xf32, #{{.*}}>
//       CHECK:   %[[D14:.*]] = memref.subview %[[D8]]{{\[}}%{{.*}}, %{{.*}}, %{{.*}}] [1, 32, 32] [1, 1, 1] : memref<10x768x2048xf32> to memref<1x32x32xf32, #{{.*}}>
//       CHECK:   gpu.barrier
//       CHECK:   %[[D15:.*]] = affine.apply #map5(){{\[}}%[[D0]], %[[D1]], %[[D2]]]
//       CHECK:   %[[D16:.*]] = affine.apply #map6(){{\[}}%[[D0]]]
//       CHECK:   %[[D17:.*]] = vector.transfer_read %[[D13]]{{\[}}%[[C0]], %[[D15]], %[[D16]]], %[[CST]] {in_bounds = [true]} : memref<1x32x32xf32, #{{.*}}>, vector<4xf32>
//       CHECK:   vector.transfer_write %[[D17]], %[[D4]]{{\[}}%[[C0]], %[[D15]], %[[D16]]] {in_bounds = [true]} : vector<4xf32>, memref<1x32x32xf32, #{{.*}}, 3>
//       CHECK:   %[[D18:.*]] = vector.transfer_read %[[D14]]{{\[}}%[[C0]], %[[D15]], %[[D16]]], %[[CST]] {in_bounds = [true]} : memref<1x32x32xf32, #{{.*}}>, vector<4xf32>
//       CHECK:   vector.transfer_write %[[D18]], %[[D6]]{{\[}}%[[C0]], %[[D15]], %[[D16]]] {in_bounds = [true]} : vector<4xf32>, memref<1x32x32xf32, #{{.*}}, 3>
//       CHECK:   gpu.barrier
//       CHECK:   %[[D19:.*]] = affine.apply #map7(){{\[}}%[[D0]]]
//       CHECK:   %[[D20:.*]] = memref.subview %[[D4]][0, %[[D19]], %[[D1]]] [1, 4, 1] [1, 1, 1] : memref<1x32x32xf32, #{{.*}}, 3> to memref<1x4x1xf32, #{{.*}}, 3>
//       CHECK:   %[[D21:.*]] = memref.subview %[[D6]][0, %[[D19]], %[[D1]]] [1, 4, 1] [1, 1, 1] : memref<1x32x32xf32, #{{.*}}, 3> to memref<1x4x1xf32, #{{.*}}, 3>
//       CHECK:   %[[D22:.*]] = memref.subview %[[D12]][0, %[[D1]], %[[D19]]] [1, 1, 4] [1, 1, 1] : memref<1x32x32xf32, #{{.*}}> to memref<1x1x4xf32, #{{.*}}>
//       CHECK:   %[[D23:.*]] = vector.transfer_read %[[D20]]{{\[}}%[[C0]], %[[C0]], %[[C0]]], %[[CST]] {in_bounds = [true, true]} : memref<1x4x1xf32, #{{.*}}, 3>, vector<4x1xf32>
//       CHECK:   %[[D24:.*]] = vector.transfer_read %[[D21]]{{\[}}%[[C0]], %[[C0]], %[[C0]]], %[[CST]] {in_bounds = [true, true]} : memref<1x4x1xf32, #{{.*}}, 3>, vector<4x1xf32>
//       CHECK:   %[[D25:.*]] = arith.addf %[[D23]], %[[D24]] : vector<4x1xf32>
//       CHECK:   %[[D26:.*]] = vector.broadcast %[[D25]] : vector<4x1xf32> to vector<1x4x1xf32>
//       CHECK:   %[[D27:.*]] = vector.shape_cast %[[D26]] : vector<1x4x1xf32> to vector<1x1x4xf32>
//       CHECK:   %[[D28:.*]] = vector.extract %[[D27]][0, 0] : vector<1x1x4xf32>
//       CHECK:   vector.transfer_write %[[D28]], %[[D22]]{{\[}}%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true]} : vector<4xf32>, memref<1x1x4xf32, #{{.*}}>
