// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-codegen-rocdl-configuration-pipeline, iree-codegen-linalg-to-rocdl-pipeline2)))" %s | FileCheck %s

hal.executable private @warp_reduction {
  hal.executable.variant @rocm target(<"rocm", "rocm-hsaco-fb", {target_arch = "gfx940"}>) {
    hal.executable.export public @warp_reduction ordinal(0) layout(
      #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @warp_reduction() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x512xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2x512xf32>> -> tensor<2x512xf32>
        %3 = tensor.empty() : tensor<2xf32>
        %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2xf32>) -> tensor<2xf32>
        %5 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]
        } ins(%2 : tensor<2x512xf32>) outs(%4 : tensor<2xf32>) {
        ^bb0(%arg0: f32, %arg1: f32):
          %6 = arith.addf %arg1, %arg0 : f32
          linalg.yield %6 : f32
        } -> tensor<2xf32>
        flow.dispatch.tensor.store %5, %1, offsets = [0], sizes = [2], strides = [1] : tensor<2xf32> -> !flow.dispatch.tensor<writeonly:tensor<2xf32>>
        return
      }
    }
  }
}

//   CHECK-LABEL: llvm.func @warp_reduction
// CHECK-COUNT-8:   rocdl.ds_bpermute

// -----

hal.executable public @main_dispatch_517 {
  hal.executable.variant public @rocm target(<"rocm", "rocm-hsaco-fb", {target_arch = "gfx90a"}>) {
    hal.executable.export public @warp_reduction_large_vector ordinal(0) layout(
      #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @warp_reduction_large_vector() {
        %cst = arith.constant 0.000000e+00 : f32
        %c128 = arith.constant 128 : index
        %c0 = arith.constant 0 : index
        %c394240 = arith.constant 394240 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c128) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x1280xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1280x1280xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c394240) : !flow.dispatch.tensor<writeonly:tensor<1x1280xf32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x1280xf32>> -> tensor<1x1280xf32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [1280, 1280], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1280x1280xf32>> -> tensor<1280x1280xf32>
        %5 = tensor.empty() : tensor<1x1280xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x1280xf32>) -> tensor<1x1280xf32>
        %7 = linalg.matmul_transpose_b ins(%3, %4 : tensor<1x1280xf32>, tensor<1280x1280xf32>) outs(%6 : tensor<1x1280xf32>) -> tensor<1x1280xf32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [1, 1280], strides = [1, 1] : tensor<1x1280xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x1280xf32>>
        return
      }
    }
  }
}

// Each workgroup (5x64 threads) handles a shape of 64x1280 (parallel x reduction).
// So we are seeing 64x(5x2) = 640 warp operations.
// TODO: we probably need to revisit the configuration heuristics here.

//     CHECK-LABEL: llvm.func @warp_reduction_large_vector
// CHECK-COUNT-640:   rocdl.ds_bpermute
