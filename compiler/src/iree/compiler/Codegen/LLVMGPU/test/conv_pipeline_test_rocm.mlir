// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx1100 \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-llvmgpu-select-lowering-strategy, func.func(iree-llvmgpu-lower-executable-target,canonicalize)))))' \
// RUN:   %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>,
    #hal.descriptor_set.binding<3, storage_buffer>
  ]>
]>
hal.executable private @conv_nchw_dispatch_1 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @conv_2d_nchw_fchw_2x320x64x64x320x3x3_f16 ordinal(0) layout(#pipeline_layout) attributes {
      translation_info = #iree_codegen.translation_info<LLVMGPUVectorize workgroup_size = [16, 2, 1]>
    } {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_2d_nchw_fchw_2x320x64x64x320x3x3_f16() {
        %cst = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2x320x130x130xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320x320x3x3xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<320xf16>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(3) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x320x64x64xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 320, 130, 130], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x320x130x130xf16>> -> tensor<2x320x130x130xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [320, 320, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<320x320x3x3xf16>> -> tensor<320x320x3x3xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0], sizes = [320], strides = [1] : !flow.dispatch.tensor<readonly:tensor<320xf16>> -> tensor<320xf16>
        %7 = tensor.empty() : tensor<2x320x64x64xf16>
        %8 = linalg.fill {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 8, 64, 4, 1, 1], [0, 0, 1, 0]]>} ins(%cst : f16) outs(%7 : tensor<2x320x64x64xf16>) -> tensor<2x320x64x64xf16>
        %9 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 8, 64, 4, 1, 1], [0, 0, 1, 0]]>, strides = dense<2> : vector<2xi64>} ins(%4, %5 : tensor<2x320x130x130xf16>, tensor<320x320x3x3xf16>) outs(%8 : tensor<2x320x64x64xf16>) -> tensor<2x320x64x64xf16>
        %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9, %6 : tensor<2x320x64x64xf16>, tensor<320xf16>) outs(%7 : tensor<2x320x64x64xf16>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 1, 8, 64, 4, 1, 1], [0, 0, 1, 0]]>} {
        ^bb0(%in: f16, %in_0: f16, %out: f16):
          %11 = arith.addf %in, %in_0 : f16
          linalg.yield %11 : f16
        } -> tensor<2x320x64x64xf16>
        flow.dispatch.tensor.store %10, %3, offsets = [0, 0, 0, 0], sizes = [2, 320, 64, 64], strides = [1, 1, 1, 1] : tensor<2x320x64x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<2x320x64x64xf16>>
        return
      }
    }
  }
}

// TODO: This test reflects a bug related to how the convolution is bufferized
// for the LLVMGPUVectorize pipeline, meaning these local memory allocations are
// not desired. This test should be dropped once the extra buffers have been
// eliminated.

//   CHECK-LABEL:  func @conv_2d_nchw_fchw_2x320x64x64x320x3x3_f16
// CHECK-COUNT-3:    memref.alloc() : memref<1x1x1x4xf16, #gpu.address_space<private>>
// CHECK-COUNT-3:    memref.copy %{{.*}}, %{{.*}} : memref<1x1x1x4xf16, #gpu.address_space<private>> to memref<{{.*}} #hal.descriptor_type<storage_buffer>>
