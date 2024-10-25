// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx940 --pass-pipeline="builtin.module(func.func(iree-llvmgpu-convolution-to-igemm),canonicalize,cse)" %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 1, 4, 32], [0, 1, 2, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]]>
func.func public @conv_with_lowering_config(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>) -> tensor<1x14x14x16xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x14x14x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  %0 = linalg.conv_2d_nhwc_hwcf {lowering_config = #config,
      dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%arg0, %arg1: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
    outs(%fill: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  return %0 : tensor<1x14x14x16xf32>
}
// CHECK:      func.func public @conv_with_lowering_config
// CHECK-NOT:    iree_linalg_ext.im2col
// CHECK:        linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:     lowering_config

// -----

func.func public @set_lowering_config(%arg0: tensor<1x34x34x128xf32>, %arg1: tensor<3x3x128x128xf32>) -> tensor<1x32x32x128xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x32x32x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x32x32x128xf32>) -> tensor<1x32x32x128xf32>
  %0 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%arg0, %arg1: tensor<1x34x34x128xf32>, tensor<3x3x128x128xf32>)
    outs(%fill: tensor<1x32x32x128xf32>) -> tensor<1x32x32x128xf32>
  return %0 : tensor<1x32x32x128xf32>
}
// CHECK:      func.func public @set_lowering_config
// CHECK:        iree_linalg_ext.im2col
// CHECK:        linalg.generic
// CHECK-SAME:     lowering_config = #iree_gpu.lowering_config<
// CHECK-SAME:         {mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>,
// CHECK-SAME:          promote_operands = [0, 1], reduction = [0, 0, 0, 0, 8],
// CHECK-SAME:          subgroup = [0, 0, 2, 2, 0], workgroup = [1, 1, 2, 8, 0]}>
