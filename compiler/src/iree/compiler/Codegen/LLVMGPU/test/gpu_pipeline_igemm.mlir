// RUN: iree-opt --pass-pipeline="builtin.module(iree-codegen-llvmgpu-configuration-pipeline)" \
// RUN:   --iree-codegen-llvmgpu-use-igemm --iree-gpu-test-target=gfx940 --split-input-file %s | FileCheck %s

// RUN: iree-opt --pass-pipeline="builtin.module(iree-codegen-rocdl-configuration-pipeline)" \
// RUN:   --iree-codegen-llvmgpu-use-igemm --iree-gpu-test-target=gfx940 --split-input-file %s | FileCheck %s

// Make sure that the GPU configuration pipelines set correct translation info for igemm.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, Indirect>
]>

func.func @nhwc_conv() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant dense<1.0> : tensor<1x64xf32>
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<1x16x16x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x3x4x16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<1x14x14x16xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 16, 16, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x16x16x4xf32>> -> tensor<1x16x16x4xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 4, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<3x3x4x16xf32>> -> tensor<3x3x4x16xf32>
  %empty = tensor.empty() : tensor<1x14x14x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  %5 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%3, %4: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
    outs(%fill: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  flow.dispatch.tensor.store %5, %2, offsets = [0, 0, 0, 0], sizes = [1, 14, 14, 16], strides = [1, 1, 1, 1] : tensor<1x14x14x16xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x14x14x16xf32>>
  return
}
// CHECK:       #[[$TRANSLATION_INFO:.+]] = #iree_codegen.translation_info
// CHECK-SAME:      LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @nhwc_conv
// CHECK-SAME:      translation_info = #[[$TRANSLATION_INFO]]
// CHECK:         iree_linalg_ext.im2col

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, Indirect>
]>

func.func @nchw_conv() {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant dense<1.0> : tensor<1x64xf32>
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<1x4x16x16xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x4x3x3xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<1x16x14x14xf32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 4, 16, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x4x16x16xf32>> -> tensor<1x4x16x16xf32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [16, 4, 3, 3], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<16x4x3x3xf32>> -> tensor<16x4x3x3xf32>
  %empty = tensor.empty() : tensor<1x16x14x14xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32>
  %5 = linalg.conv_2d_nchw_fchw
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%3, %4: tensor<1x4x16x16xf32>, tensor<16x4x3x3xf32>)
    outs(%fill: tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32>
  flow.dispatch.tensor.store %5, %2, offsets = [0, 0, 0, 0], sizes = [1, 16, 14, 14], strides = [1, 1, 1, 1] : tensor<1x16x14x14xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x16x14x14xf32>>
  return
}
// CHECK:       #[[$TRANSLATION_INFO:.+]] = #iree_codegen.translation_info
// CHECK-SAME:      LLVMGPUTileAndFuse workgroup_size = [64, 1, 1] subgroup_size = 64
// CHECK-LABEL: func.func @nchw_conv
// CHECK-SAME:      translation_info = #[[$TRANSLATION_INFO]]
// CHECK:         iree_linalg_ext.im2col
