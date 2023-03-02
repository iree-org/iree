// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' %s | FileCheck %s

#device_target_cuda = #hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}>]}>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>
module attributes {hal.device.targets = [#device_target_cuda]} {
  hal.executable private @conv2d_1x230x230x3_7x7x3x64_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {
      hal.executable.export public @conv2d_1x230x230x3_7x7x3x64 ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @conv2d_1x230x230x3_7x7x3x64() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<1x230x230x3xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<7x7x3x64xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x112x112x64xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 230, 230, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x230x230x3xf32>> -> tensor<1x230x230x3xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [7, 7, 3, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<7x7x3x64xf32>> -> tensor<7x7x3x64xf32>
          %5 = tensor.empty() : tensor<1x112x112x64xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
          %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x230x230x3xf32>, tensor<7x7x3x64xf32>) outs(%6 : tensor<1x112x112x64xf32>) -> tensor<1x112x112x64xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [1, 112, 112, 64], strides = [1, 1, 1, 1] : tensor<1x112x112x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x112x112x64xf32>>
          return
        }
      }
    }
  }
}

//   CHECK-LABEL:  func.func @conv2d_1x230x230x3_7x7x3x64
//     CHECK-NOT:    vector.transfer_write
//     CHECK-NOT:    vector.transfer_read
//         CHECK:    scf.for
//         CHECK:      scf.for
// CHECK-COUNT-2:        vector.transfer_read
// CHECK-COUNT-4:        vector.contract
//         CHECK:      scf.yield %{{.*}} : vector<1x4x4xf32>
//         CHECK:    scf.yield %{{.*}} : vector<1x4x4xf32>
//         CHECK:    vector.transfer_write {{.*}} : vector<4x4xf32>, memref<1x112x112x64xf32>

// -----

#device_target_cuda = #hal.device.target<"cuda", {executable_targets = [#hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}>], legacy_sync}>
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>
module attributes {hal.device.targets = [#device_target_cuda]} {
  hal.executable private @conv_nchw_dispatch_0 {
    hal.executable.variant public @cuda_nvptx_fb, target = #executable_target_cuda_nvptx_fb {
      hal.executable.export public @conv_nchw ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index):
        %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @conv_nchw() {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x4x66x66xf32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<320x4x3x3xf32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x320x64x64xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 230, 230, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x4x66x66xf32>> -> tensor<2x4x66x66xf32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [7, 7, 3, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<320x4x3x3xf32>> -> tensor<320x4x3x3xf32>
          %5 = tensor.empty() : tensor<2x320x64x64xf32>
          %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x320x64x64xf32>) -> tensor<2x320x64x64xf32>
          %7 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
            ins(%3, %4 : tensor<2x4x66x66xf32>, tensor<320x4x3x3xf32>)
            outs(%6 : tensor<2x320x64x64xf32>) -> tensor<2x320x64x64xf32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 320, 6, 64], strides = [1, 1, 1, 1] : tensor<2x320x64x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<2x320x64x64xf32>>
          return
        }
      }
    }
  }
}

//   CHECK-LABEL:  func.func @conv_nchw
// TODO: hoist the accumulator read and fold the transfer_write.
//         CHECK:    vector.transfer_write
//         CHECK:    scf.for
//         CHECK:      scf.for
// CHECK-COUNT-3:        vector.transfer_read
//         CHECK:        vector.contract
//         CHECK:        vector.transfer_write
// CHECK-COUNT-2:        vector.transfer_read
//         CHECK:        vector.contract
//         CHECK:        vector.transfer_write
// CHECK-COUNT-2:        vector.transfer_read
//         CHECK:        vector.contract
//         CHECK:        vector.transfer_write
// CHECK-COUNT-2:        vector.transfer_read
//         CHECK:        vector.contract
//         CHECK:        vector.transfer_write
