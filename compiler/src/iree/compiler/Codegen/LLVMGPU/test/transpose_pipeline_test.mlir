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
//       CHECK:  %[[R1:.+]] = vector.transfer_read %[[ALLOC]][%{{.*}}, %{{.*}}], %[[CST]] {in_bounds = [true, true]} : memref<32x33xf32, 3>, vector<4x1xf32>
//       CHECK:  %[[R2:.+]] = vector.shape_cast %[[R1]] : vector<4x1xf32> to vector<1x4xf32>
//       CHECK:  %[[R3:.+]] = vector.transfer_read %[[IN1]][%{{.*}}, %{{.*}}], %[[CST]] {in_bounds = [true]} : memref<768x2048xf32>, vector<4xf32>
//       CHECK:  %[[R4:.+]] = vector.extract %[[R2]][0] : vector<1x4xf32>
//       CHECK:  %[[R5:.+]] = arith.addf %[[R4]], %[[R3]] : vector<4xf32>
//       CHECK:  vector.transfer_write %[[R5]], %[[OUT]][%{{.*}}, %{{.*}}] {in_bounds = [true]} : vector<4xf32>, memref<768x2048xf32>
