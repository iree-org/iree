// RUN: iree-opt --split-input-file --pass-pipeline='hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target-pass))' %s | FileCheck %s

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
//       CHECK:  %[[D3:.+]] = memref.alloc() : memref<32x33xf32, 3>
//       CHECK:  %[[D4:.+]] = memref.subview %[[D3]][0, 0] [32, 32] [1, 1] : memref<32x33xf32, 3> to memref<32x32xf32, #{{.*}}, 3>
//       CHECK:  %[[D9:.+]] = memref.subview %[[D6:.+]][%{{.*}}, %{{.*}}] [32, 32] [1, 1] : memref<4096x4096xf32> to memref<32x32xf32, #{{.*}}>
//       CHECK:  %[[D10:.+]] = memref.subview %[[D5:.+]][%{{.*}}, %{{.*}}] [32, 32] [1, 1] : memref<4096x4096xf32> to memref<32x32xf32, #{{.*}}>
//       CHECK:  gpu.barrier
//       CHECK:  %[[D13:.+]] = vector.transfer_read %[[D10]][%{{.*}}, %{{.*}}], %[[CST]] {in_bounds = [true]} : memref<32x32xf32, #{{.*}}>, vector<4xf32>
//       CHECK:  vector.transfer_write %[[D13]], %[[D4]][%{{.*}}, %{{.*}}] {in_bounds = [true]} : vector<4xf32>, memref<32x32xf32, #{{.*}}, 3>
//       CHECK:  gpu.barrier
//       CHECK:  %[[D15:.+]] = memref.subview %[[D4]][%{{.*}}, %{{.*}}] [4, 1] [1, 1] : memref<32x32xf32, #{{.*}}, 3> to memref<4x1xf32, #{{.*}}, 3>
//       CHECK:  %[[D16:.+]] = memref.subview %[[D9]][%{{.*}}, %{{.*}}] [1, 4] [1, 1] : memref<32x32xf32, #{{.*}}> to memref<1x4xf32, #{{.*}}>
//       CHECK:  %[[D17:.+]] = vector.transfer_read %[[D15]][%{{.*}}, %{{.*}}], %[[CST]] {in_bounds = [true, true]} : memref<4x1xf32, #{{.*}}, 3>, vector<4x1xf32>
//       CHECK:  %[[D18:.+]] = vector.shape_cast %[[D17]] : vector<4x1xf32> to vector<1x4xf32>
//       CHECK:  %[[D19:.+]] = vector.extract %[[D18]][0] : vector<1x4xf32>
//       CHECK:  vector.transfer_write %[[D19]], %[[D16]][%{{.*}}, %{{.*}}] {in_bounds = [true]} : vector<4xf32>, memref<1x4xf32, #{{.*}}>
// -----
