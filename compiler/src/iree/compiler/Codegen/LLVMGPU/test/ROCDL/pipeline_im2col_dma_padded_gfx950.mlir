// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx950 \
// RUN:   --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-llvmgpu-lower-executable-target{for-rocdl=true})))))" %s | FileCheck %s

// Padded convolution: input is NOT pre-padded. tensor.pad produces a
// same-padded input, and the conv produces an output of the same spatial
// dimensions. The pad is absorbed into im2col as input_pad_low /
// input_pad_high / pad_value and lowered through the DMA async-copy path
// under pad_value=0 via the fat_raw_buffer OOB sentinel.
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation = #iree_codegen.translation_info<pipeline =
  #iree_gpu.pipeline<TileAndFuse>
  workgroup_size = [512, 1, 1]
  subgroup_size = 64,
  {
     gpu_pipeline_options = #iree_gpu.pipeline_options<
       prefetch_num_stages = 2,
       no_reduce_shared_memory_bank_conflicts = true,
       use_igemm_convolution = true>
  }>
#config = #iree_gpu.lowering_config<{
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>,
  promote_operands = [0, 1],
  promotion_types = [#iree_gpu.use_global_load_dma, #iree_gpu.use_global_load_dma],
  reduction = [0, 0, 0, 0, 4],
  subgroup = [1, 2, 1, 4, 0],
  workgroup = [1, 4, 32, 128, 0]
}>
hal.executable private @conv_im2col_dma_padded {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @conv_im2col_dma_padded ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_im2col_dma_padded() attributes {translation_info = #translation} {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_f16 = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x32x32x1280xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x1280x1280xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 32, 32, 1280], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x32x32x1280xf16>> -> tensor<2x32x32x1280xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 1280, 1280], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x1280x1280xf16>> -> tensor<3x3x1280x1280xf16>
        %padded = tensor.pad %3 low[0, 1, 1, 0] high[0, 1, 1, 0] {
        ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
          tensor.yield %cst_f16 : f16
        } : tensor<2x32x32x1280xf16> to tensor<2x34x34x1280xf16>
        %5 = tensor.empty() : tensor<2x32x32x1280xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, lowering_config = #config} ins(%padded, %4 : tensor<2x34x34x1280xf16>, tensor<3x3x1280x1280xf16>) outs(%6 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 32, 32, 1280], strides = [1, 1, 1, 1] : tensor<2x32x32x1280xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>>
        return
      }
    }
  }
}

// Verify padded im2col DMA path: conv is lowered through im2col -> gather
// -> coalesced_gather_dma -> amdgpu.gather_to_lds, with the fat_raw_buffer
// OOB-to-zero behavior handling the pad positions (no tensor.pad in the
// lowered IR).
//
//    CHECK-LABEL: func @conv_im2col_dma_padded
//      CHECK-NOT:   tensor.pad
//          CHECK:   scf.forall
//          CHECK:     scf.for {{.*}} iter_args
//          CHECK:       amdgpu.gather_to_lds {{.*}}#amdgpu.address_space<fat_raw_buffer>{{.*}}#gpu.address_space<workgroup>
//          CHECK:       gpu.barrier
//          CHECK:       amdgpu.mfma 16x16x32
//          CHECK:       scf.yield

// -----

// 5x5 same-padding variant: pad=2 on H/W, kernel 5x5. Exercises more OOB
// positions per window than the 3x3 case and keeps the same output spatial
// shape so the workgroup/subgroup tile config is reused unchanged.
#pipeline_layout_5x5 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#translation_5x5 = #iree_codegen.translation_info<pipeline =
  #iree_gpu.pipeline<TileAndFuse>
  workgroup_size = [512, 1, 1]
  subgroup_size = 64,
  {
     gpu_pipeline_options = #iree_gpu.pipeline_options<
       prefetch_num_stages = 2,
       no_reduce_shared_memory_bank_conflicts = true,
       use_igemm_convolution = true>
  }>
#config_5x5 = #iree_gpu.lowering_config<{
  mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x32_F16>,
  promote_operands = [0, 1],
  promotion_types = [#iree_gpu.use_global_load_dma, #iree_gpu.use_global_load_dma],
  reduction = [0, 0, 0, 0, 4],
  subgroup = [1, 2, 1, 4, 0],
  workgroup = [1, 4, 32, 128, 0]
}>
hal.executable private @conv_im2col_dma_padded_5x5 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb">) {
    hal.executable.export public @conv_im2col_dma_padded_5x5 ordinal(0) layout(#pipeline_layout_5x5) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @conv_im2col_dma_padded_5x5() attributes {translation_info = #translation_5x5} {
        %cst = arith.constant 0.000000e+00 : f32
        %cst_f16 = arith.constant 0.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout_5x5) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x32x32x1280xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout_5x5) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x5x1280x1280xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout_5x5) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 32, 32, 1280], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x32x32x1280xf16>> -> tensor<2x32x32x1280xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [5, 5, 1280, 1280], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<5x5x1280x1280xf16>> -> tensor<5x5x1280x1280xf16>
        %padded = tensor.pad %3 low[0, 2, 2, 0] high[0, 2, 2, 0] {
        ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
          tensor.yield %cst_f16 : f16
        } : tensor<2x32x32x1280xf16> to tensor<2x36x36x1280xf16>
        %5 = tensor.empty() : tensor<2x32x32x1280xf32>
        %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
        %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, lowering_config = #config_5x5} ins(%padded, %4 : tensor<2x36x36x1280xf16>, tensor<5x5x1280x1280xf16>) outs(%6 : tensor<2x32x32x1280xf32>) -> tensor<2x32x32x1280xf32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 32, 32, 1280], strides = [1, 1, 1, 1] : tensor<2x32x32x1280xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32x32x1280xf32>>
        return
      }
    }
  }
}

//    CHECK-LABEL: func @conv_im2col_dma_padded_5x5
//      CHECK-NOT:   tensor.pad
//          CHECK:   scf.forall
//          CHECK:     scf.for {{.*}} iter_args
//          CHECK:       amdgpu.gather_to_lds {{.*}}#amdgpu.address_space<fat_raw_buffer>{{.*}}#gpu.address_space<workgroup>
//          CHECK:       gpu.barrier
//          CHECK:       amdgpu.mfma 16x16x32
//          CHECK:       scf.yield
