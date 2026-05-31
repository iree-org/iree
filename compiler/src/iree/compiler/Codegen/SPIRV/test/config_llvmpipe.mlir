// RUN: iree-opt --split-input-file --iree-gpu-test-target=llvmpipe@vulkan --pass-pipeline='builtin.module(func.func(iree-codegen-gpu-generalize-named-ops),iree-spirv-select-lowering-strategy-pass)' %s | FileCheck %s

// `llvmpipe` is a software (CPU) Vulkan implementation without vendor-specific
// codegen, so it falls back to the default SPIR-V tile-and-vectorize path. These
// tests check that the target resolves and produces a valid lowering strategy.

func.func @matmul_1024x2048x512(%3: tensor<1024x512xf32>, %4: tensor<512x2048xf32>) -> tensor<1024x2048xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1024x2048xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1024x2048xf32>) -> tensor<1024x2048xf32>
  %7 = linalg.matmul ins(%3, %4 : tensor<1024x512xf32>, tensor<512x2048xf32>) outs(%6 : tensor<1024x2048xf32>) -> tensor<1024x2048xf32>
  return %7 : tensor<1024x2048xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[16, 256], [8, 8], [0, 0, 4]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<BaseVectorize> workgroup_size = [32, 2, 1]>
//      CHECK: func.func @matmul_1024x2048x512(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.generic
// CHECK-SAME:       lowering_config = #[[CONFIG]]

// -----

func.func @conv_112x112x512(%3: tensor<1x225x225x3xf32>, %4: tensor<3x3x3x512xf32>) -> tensor<1x112x112x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : tensor<1x112x112x512xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x112x112x512xf32>) -> tensor<1x112x112x512xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%3, %4 : tensor<1x225x225x3xf32>, tensor<3x3x3x512xf32>) outs(%6 : tensor<1x112x112x512xf32>) -> tensor<1x112x112x512xf32>
  return %7 : tensor<1x112x112x512xf32>
}

//  CHECK-DAG: #[[CONFIG:.+]] = #iree_codegen.lowering_config<tile_sizes = {{\[}}[0, 1, 8, 128], [1, 1, 8, 4], [0, 0, 0, 0, 1, 1, 4], [0, 1, 0, 0]{{\]}}>
//  CHECK-DAG: #[[TRANSLATION:.+]] = #iree_codegen.translation_info<pipeline = #iree_gpu.spirv_pipeline<BaseVectorize> workgroup_size = [32, 1, 1]>
//      CHECK: func.func @conv_112x112x512(
// CHECK-SAME:     translation_info = #[[TRANSLATION]]
//      CHECK:   linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:       lowering_config = #[[CONFIG]]
