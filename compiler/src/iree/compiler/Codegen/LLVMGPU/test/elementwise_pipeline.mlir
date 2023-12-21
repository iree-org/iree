// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-select-lowering-strategy, iree-llvmgpu-lower-executable-target)))" %s | FileCheck %s

hal.executable @warp_reduction_dispatch {
hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>) {
  hal.executable.export public @forward_dispatch_0_generic_320x320x3x3 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index, %arg4: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3, %arg4
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @forward_dispatch_0_generic_320x320x3x3() {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x320x320x3xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<320x320x3x3xf32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [3, 320, 320, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x320x320x3xf32>> -> tensor<3x320x320x3xf32>
      %3 = tensor.empty() : tensor<320x320x3x3xf32>
      %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<3x320x320x3xf32>) outs(%3 : tensor<320x320x3x3xf32>) {
      ^bb0(%in: f32, %out: f32):
        %5 = arith.addf %in, %cst : f32
        linalg.yield %5 : f32
      } -> tensor<320x320x3x3xf32>
      flow.dispatch.tensor.store %4, %1, offsets = [0, 0, 0, 0], sizes = [320, 320, 3, 3], strides = [1, 1, 1, 1] : tensor<320x320x3x3xf32> -> !flow.dispatch.tensor<writeonly:tensor<320x320x3x3xf32>>
      return
    }
  }
}
}

// CHECK-LABEL: hal.executable.export public @forward_dispatch_0_generic_320x320x3x3
//     CHECK:     workgroup_size = [3 : index, 3 : index, 7 : index]}
// CHECK-DAG:     %[[C46:.+]] = arith.constant 46 : index
// CHECK-DAG:     %[[C320:.+]] = arith.constant 320 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
//     CHECK:     hal.return %[[C46]], %[[C320]], %[[C1]] : index, index, index
